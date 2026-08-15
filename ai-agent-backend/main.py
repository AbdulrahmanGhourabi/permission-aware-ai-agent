import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid

import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

from ingest import extract_text, chunk_text, embed_chunks

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL")

RETRIEVAL_DEBUG = os.getenv("RETRIEVAL_DEBUG", "0") == "1"

logger = logging.getLogger("retrieval_debug")
if RETRIEVAL_DEBUG and not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

generation_logger = logging.getLogger("generation")
generation_logger.setLevel(logging.WARNING)

ROUTER_MODEL = "llama-3.1-8b-instant"
ANSWER_MODEL = "llama-3.3-70b-versatile"
DECOMPOSE_MODEL = "llama-3.3-70b-versatile"
DECOMPOSITION_ENABLED = True
ROUTING_MAX_ATTEMPTS = 2
MAX_GENERATION_RETRIES = 2
MAX_CHUNK_DISTANCE = 1.0
CANDIDATE_POOL_SIZE = 40
RERANK_TOP_K = 8
MIN_RERANK_SCORE = -9.0
ALWAYS_KEEP_TOP_CANDIDATE = True

NO_CONTEXT_ANSWER = "I don't have access to any documents that answer this."

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Permission-Aware AI API")
security = HTTPBearer()

_jwk_client = jwt.PyJWKClient(SUPABASE_JWKS_URL) if SUPABASE_JWKS_URL else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if _jwk_client is None:
        raise HTTPException(status_code=500, detail="SUPABASE_JWKS_URL not configured")

    token = credentials.credentials
    try:
        signing_key = _jwk_client.get_signing_key_from_jwt(token).key
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["ES256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    return payload


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def get_or_create_user(email: str, google_id: str | None = None, role: str = "member") -> str:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        if row:
            return str(row[0])

        user_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO users (id, email, role, google_id) VALUES (%s, %s, %s, %s)",
            (user_id, email, role, google_id),
        )
        conn.commit()
        return user_id
    finally:
        conn.close()


def get_user_role(user_id: str) -> str:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return row[0] if row else "member"
    finally:
        conn.close()


def require_document_owner(cur, document_id: str, requester_id: str) -> None:
    """Raise 404 if requester has no access, 403 if they're not the owner.
    Used by endpoints that mutate a document's sharing state.
    """
    cur.execute(
        """
        SELECT dp.permission FROM document_permissions dp
        JOIN documents d ON d.id = dp.document_id
        WHERE dp.document_id = %s AND dp.user_id = %s AND d.is_current = true
        """,
        (document_id, requester_id),
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found or you don't have access to it.",
        )
    if row[0] != "owner":
        raise HTTPException(
            status_code=403,
            detail="Only the document owner can do this.",
        )


def query_internal_records(record_type: str, user_role: str) -> list:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT data FROM internal_records
            WHERE type = %s AND %s = ANY(allowed_roles)
            """,
            (record_type, user_role),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


class RetrievedChunk:
    __slots__ = ("content", "title", "document_id")

    def __init__(self, content: str, title: str, document_id: str):
        self.content = content
        self.title = title
        self.document_id = document_id

    def as_prompt_block(self) -> str:
        return f"[Source: {self.title}]\n{self.content}"


def search_chunks_for_user(
    question: str,
    user_id: str,
    top_k: int = RERANK_TOP_K,
    max_distance: float = MAX_CHUNK_DISTANCE,
) -> list[RetrievedChunk]:
    question_embedding = embedding_model.encode(question).tolist()

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.content, c.document_id, d.title, c.embedding <=> %s::vector AS distance
            FROM chunks c
            JOIN document_permissions dp ON dp.document_id = c.document_id
            JOIN documents d ON d.id = c.document_id
            WHERE dp.user_id = %s AND d.is_current = true
            ORDER BY distance ASC, c.id ASC
            LIMIT %s
            """,
            (question_embedding, user_id, CANDIDATE_POOL_SIZE),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if RETRIEVAL_DEBUG:
        logger.info("=== RETRIEVAL_DEBUG: question=%r ===", question)
        logger.info(
            "-- stage 1: raw vector search, top %d by distance (before max_distance=%.2f filter) --",
            CANDIDATE_POOL_SIZE, max_distance,
        )
        for content, document_id, title, distance in rows:
            kept = "KEPT" if distance <= max_distance else "CUT (max_distance)"
            logger.info(
                "  [%s] dist=%.4f title=%r content=%r",
                kept, distance, title, content[:80].replace("\n", " ") + ("..." if len(content) > 80 else ""),
            )

    candidates = [
        RetrievedChunk(content=content, document_id=str(document_id), title=title)
        for content, document_id, title, distance in rows
        if distance <= max_distance
    ]
    if not candidates:
        if RETRIEVAL_DEBUG:
            logger.info("-- result: 0 candidates survived max_distance filter, returning [] --")
        return []

    pairs = [(question, c.as_prompt_block()) for c in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    if RETRIEVAL_DEBUG:
        logger.info(
            "-- stage 2: cross-encoder rerank, %d candidates (before MIN_RERANK_SCORE=%.2f filter) --",
            len(scored), MIN_RERANK_SCORE,
        )
        for chunk, score in scored:
            kept = "KEPT" if score >= MIN_RERANK_SCORE else "CUT (min_rerank_score)"
            logger.info("  [%s] score=%.4f title=%r", kept, float(score), chunk.title)

    relevant = [chunk for chunk, score in scored if score >= MIN_RERANK_SCORE]

    if not relevant and scored and ALWAYS_KEEP_TOP_CANDIDATE:
        top_chunk, top_score = scored[0]
        if RETRIEVAL_DEBUG:
            logger.info(
                "-- stage 2b: all candidates below MIN_RERANK_SCORE=%.2f, "
                "ALWAYS_KEEP_TOP_CANDIDATE=True -> keeping top scorer anyway: "
                "score=%.4f title=%r --",
                MIN_RERANK_SCORE, float(top_score), top_chunk.title,
            )
        relevant = [top_chunk]

    if RETRIEVAL_DEBUG:
        final = relevant[:top_k]
        logger.info(
            "-- stage 3: final top_k=%d returned to caller: %s --",
            top_k, [c.title for c in final] if final else "[] (0 chunks - NO_CONTEXT_ANSWER will be used)",
        )

    return relevant[:top_k]


DECOMPOSE_SYSTEM_PROMPT = (
    "You split a user's question into separate, independent sub-questions "
    "if and only if it genuinely asks about more than one distinct TOPIC. "
    "Most questions are single-topic - do NOT split those, just return the "
    "original question unchanged as the only item in the array.\n\n"
    "Split ONLY when the question explicitly signals a second, unrelated "
    "topic - look for phrases like 'and separately', 'also confirm', 'in "
    "addition', or a clear topic change mid-question (e.g. switching from "
    "asking about one policy area to a completely different policy area).\n\n"
    "CRITICAL: multiple ATTRIBUTES of the same single topic are NOT "
    "separate sub-questions. A question asking for a company's name, a "
    "person's name, AND a Slack handle - all about the same "
    "on-call/escalation lookup - is ONE topic with several requested "
    "fields, not multiple topics. Do not split a request for several "
    "fields (name + role + handle + tool, etc.) about the same thing.\n\n"
    "Do NOT split a single comparative question ('compare X and Y', 'is A "
    "the same as B').\n\n"
    "Worked example - INPUT: \"For every company whose Engineering team's "
    "on-call tool is PagerDuty, list the company, the lead's name, and "
    "their Slack handle - and separately confirm which company's "
    "remote-work notice period is also 24 hours, making clear that number "
    "is unrelated to any incident SLA figures.\" This has TWO topics "
    "(marked by 'and separately confirm'): (1) the PagerDuty/lead/Slack "
    "lookup - company, lead name, and Slack handle are three FIELDS of "
    "this ONE topic, not three sub-questions - and (2) the 24-hour "
    "remote-work-notice-vs-SLA topic. CORRECT split:\n"
    "[\"For every company whose Engineering team's on-call tool is "
    "PagerDuty, list the company, the lead's name, and their Slack "
    "handle\", \"Which company's remote-work notice period is 24 hours, "
    "and confirm that is unrelated to any incident SLA figures\"]\n"
    "WRONG split (never do this): splitting the first topic's own "
    "requested fields (name vs Slack handle) into separate items, or "
    "dropping the second topic entirely.\n\n"
    "Return ONLY a JSON array of strings, each a self-contained "
    "sub-question that could be answered independently. No other text, no "
    "markdown formatting, no explanation."
)


def decompose_question(question: str) -> list[str]:
    if not DECOMPOSITION_ENABLED:
        return [question]
    try:
        response = groq_client.chat.completions.create(
            model=DECOMPOSE_MODEL,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        if (
            isinstance(parsed, list)
            and len(parsed) >= 1
            and all(isinstance(x, str) and x.strip() for x in parsed)
        ):
            return parsed
    except Exception:
        pass
    return [question]


_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{1,30}\b")

_PROPER_NOUN_STOPWORDS = {
    "The", "This", "That", "These", "Those", "What", "Which", "Who",
    "When", "Where", "Why", "How", "Is", "Are", "Does", "Do", "Can",
    "Please", "Find", "List", "Show", "Tell", "Compare", "Company",
    "Companies", "Engineering", "Department", "Team", "Policy", "Slack",
    "Section",
}


def sanitize_router_query(router_query: str, original_question: str) -> str:
    original_words = {w.lower() for w in _PROPER_NOUN_RE.findall(original_question)}
    router_proper_nouns = set(_PROPER_NOUN_RE.findall(router_query)) - _PROPER_NOUN_STOPWORDS

    for noun in router_proper_nouns:
        if noun.lower() not in original_words:
            if RETRIEVAL_DEBUG:
                logger.info(
                    "sanitize_router_query: rejected rewrite %r - "
                    "introduced %r not present in original question %r, "
                    "falling back to original question",
                    router_query, noun, original_question,
                )
            return original_question

    return router_query


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""

    seen_titles: dict[str, list[RetrievedChunk]] = {}
    order: list[str] = []
    for c in chunks:
        if c.title not in seen_titles:
            seen_titles[c.title] = []
            order.append(c.title)
        seen_titles[c.title].append(c)

    blocks = []
    for title in order:
        group = seen_titles[title]
        body = "\n".join(c.content for c in group)
        blocks.append(f"[Source: {title}]\n{body}")

    return "\n\n".join(blocks)


def build_document_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)
    return f"""Answer the question using ONLY the context below.

Each piece of context is labeled with its source document, like
"[Source: <document title>]". If the question names a specific company or
document, use ONLY the context labeled with that source - never use a
value from a different source's chunk to answer, even if that chunk covers
the same section or field name. If two sources contain similar-looking
information, do not merge or average them; treat them as answers to
different questions.

The context may also include text extracted from tables where columns
were flattened into sequential lines (e.g. two department columns listed
one after another, each with Lead/Slack/Tool lines beneath it). Before
concluding the answer isn't present, check whether it can be reconstructed
from the ordering and labels in the text - group each "Lead:", "Slack:",
"Tool:" etc. line with the department/person heading that most recently
preceded it, within that same source.

Before writing your final answer, first work through the context source by
source:
1. List every distinct [Source: ...] label present in the context below,
   one per line.
2. For each source you listed, write the specific value, fact, or field
   from that source that is relevant to the question - even if it turns
   out not to be needed in your final answer, and even if two sources give
   similar-looking values. Do not skip a source just because another
   source already seems to answer the question.
3. Before writing "no relevant information" for any source, re-read that
   source's full text below one more time and quote the exact phrase you
   checked against the question. Only conclude a source has no relevant
   information if you can show that quoted phrase does not address the
   question - never state "no relevant information" without quoting what
   you checked, since the answer may be present in a form you skimmed
   past the first time (e.g. a short field like "On-call is X." near the
   end of a longer chunk).
4. When quoting a field like "On-call is X" or "Escalation is Y" in your
   per-source list, ALWAYS include the department/team/entity name it
   belongs to on the SAME line, in the form "<Entity>: <field> is <value>"
   - e.g. "Engineering: On-call is PagerDuty", never a bare "On-call is
   PagerDuty" with the entity name left implicit or on a separate line.
   In your final answer, only attach a value to an entity name if that
   exact pairing appeared together in your per-source list - never
   re-pair a value with a different entity than the one it was quoted
   with, even if that entity seems like a plausible fit.
5. If your final answer is a list/enumeration (e.g. "which companies do
   X"), before writing it, count how many sources in your per-source list
   from step 2-4 actually contain a matching value for the question. Your
   final answer's list MUST include that exact same count - if step 2-4
   found 4 sources with a matching PagerDuty/escalation value, your final
   answer must name all 4, not a subset. Silently dropping one is a
   common error - double check the count matches before finalizing.
6. If the question asks you to compare two values, first check whether
   the two values are actually the same TYPE of measurement (e.g. both
   currency amounts, both day-counts, both percentages). If they measure
   different things (e.g. a dollar amount vs. a number of days), do NOT
   state which one is "higher" or "lower" - numeric comparison across
   different units is meaningless even if one number is arithmetically
   larger. In that case, your final answer must explicitly refuse the
   higher/lower comparison and explain why the two values are not
   comparable, without also asserting a direction (never say "X is higher
   than Y, but they're not really comparable" - if they're not comparable,
   do not state a direction at all).
7. Only after completing that per-source list and the count check, write
   your final answer, drawing only from the values you just listed and
   citing which source each part came from.

Steps 1-6 above are your own internal working - the user never sees them,
so their format doesn't matter. What matters is step 7.

CRITICAL OUTPUT FORMAT: after finishing steps 1-6, write the exact
literal text "===FINAL ANSWER===" on its own line, then write ONLY the
final answer after it. Everything before "===FINAL ANSWER===" is scratch
work; everything after it is shown directly to the user, so it must
stand alone as a complete, polished answer.

The text after "===FINAL ANSWER===" must:
- Read like a natural response from a helpful colleague, not a report -
  plain prose (or a short list only if the question itself asks for a
  list), no headers, no meta-commentary about "sources", "steps",
  "context", or the reasoning process itself.
- NEVER show a raw filename or a bracketed tag like "[Source:
  Acme_Corp_Employee_Handbook.pdf]". If you need to attribute a fact to
  where it came from, name the document naturally in a sentence instead -
  e.g. "According to the Acme Corp handbook, ..." or "Per Borealis's
  policy, ...". Convert a filename like "Acme_Corp_Employee_Handbook.pdf"
  into its natural name ("Acme Corp handbook", "Acme Corp's employee
  handbook") rather than reading the filename verbatim, and never include
  the file extension.
- Only name a source when it adds real value (the question involves more
  than one company/document, or the person would reasonably want to know
  which policy governs). For a simple single-fact answer from a single
  source, a plain answer with no attribution at all is fine.

If, after that check, the context still doesn't contain the answer, the
text after "===FINAL ANSWER===" must be EXACTLY this sentence and
nothing else - no explanation of what you checked, no mention of which
sources you looked at:
"I don't have access to any documents that answer this."
Do not use any knowledge from outside the context, even if a name or fact
seems familiar.

Context:
{context}

Question: {question}

Work through each source, then answer:"""


def build_records_prompt(question: str, records: list) -> str:
    context = json.dumps(records, indent=2)
    return f"""Answer the question using ONLY the structured data below.

Data:
{context}

Question: {question}

Answer:"""


def log_agent_call(user_id: str, query: str, tool_called: str, success: bool, reasoning: str = ""):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO agent_logs (id, user_id, query, tool_called, reasoning, success)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), user_id, query, tool_called, reasoning, success),
        )
        conn.commit()
    finally:
        conn.close()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the user's uploaded company documents (wikis, policies, reports). "
                "Use this for ANY question that could plausibly be answered by a company "
                "document - policies, benefits, vacation/leave, department contacts/leads, "
                "security practices, product roadmap, support process, HR rules, or anything "
                "else a company might document. If in doubt, use this tool rather than "
                "no_tool_needed - a search that finds nothing costs almost nothing, but "
                "skipping the search risks a wrong or hallucinated answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_company_data",
            "description": (
                "Query structured internal company records about THIS company's own "
                "employees or projects (headcount, who works where, project status/counts). "
                "Only use this when the question is clearly about this specific company's "
                "staff or projects. Do NOT use for general knowledge questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_type": {
                        "type": "string",
                        "enum": ["employee", "project"],
                        "description": "Which kind of record to fetch",
                    }
                },
                "required": ["record_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "no_tool_needed",
            "description": (
                "Use this ONLY for universal facts with zero dependency on any company - "
                "pure math, general history, geography, science, literature, or plain "
                "small talk. If the question could even loosely relate to a policy, "
                "benefit, process, or fact a company might document, do NOT use this - "
                "use search_documents instead."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

AGENT_SYSTEM_PROMPT = (
    "You are a workplace assistant. You MUST always call exactly one of the three "
    "available functions - there is no other way to respond, and you must never "
    "answer from your own memory.\n\n"
    "CRITICAL: You have NO real knowledge about this specific company, its name, "
    "its products, its internal systems, its policies, its employees, or its "
    "departments. Anything you seem to 'recognize' - including generic-sounding "
    "topics like vacation policy, PTO, benefits, or department leads - is NOT "
    "reliable, because THIS company's specific facts could differ from any general "
    "pattern you've seen. You must never rely on general/average/typical knowledge "
    "to answer a question that could instead be answered from this company's own "
    "documents or records.\n\n"
    "Tools:\n"
    "- search_documents: use for anything that could be covered in this company's "
    "internal documents, wikis, policies, benefits, department/team contacts, "
    "security practices, roadmap, or support process - including questions about "
    "the company's own products, systems, or internal tools by name, even if the "
    "name sounds familiar to you, AND including generic-sounding policy/benefits "
    "questions (vacation days, sick leave, tenure-based benefits, who leads a "
    "department, etc). When uncertain whether something is company-specific or "
    "universal, default to search_documents.\n"
    "- query_company_data: use for direct factual lookups about this company's own "
    "employees or projects (headcount, who works where, project status/counts).\n"
    "- no_tool_needed: use ONLY for facts that are true regardless of employer and "
    "could not possibly be governed by a company policy or document (math, general "
    "history, geography, science, literature, well-known public facts, small talk).\n\n"
    "Decision rule: if the question contains or implies ANY proper noun that could "
    "be this company's name, product, system, team, or an employee, you must use a "
    "tool - never answer from what you think you know about that name. The same "
    "applies to any policy/benefit/process/contact question, even if phrased "
    "generically - 'how many vacation days' or 'who leads Finance' sound universal "
    "but are actually determined per-company, so they must go through "
    "search_documents, not general knowledge. If a question could plausibly be "
    "interpreted either as a universal fact or as something specific to this "
    "company, always choose the company-related tool. A wrong tool call that finds "
    "nothing costs almost nothing; confidently answering from unreliable memory "
    "about someone else's company is a critical failure.\n\n"
    "Examples:\n"
    "Q: \"What is the support policy for the assistant?\" -> search_documents\n"
    "Q: \"How many vacation days do you get after 10 years?\" -> search_documents "
    "(policy/benefit question - sounds generic but is company-specific, never answer "
    "from a general industry average)\n"
    "Q: \"Who leads the Finance department?\" -> search_documents (department/contact "
    "question, never answer from a guessed or familiar-sounding name)\n"
    "Q: \"Is every tool call logged?\" -> search_documents (this describes how the "
    "system itself is documented internally, not a request for employee/project data)\n"
    "Q: \"What does Sentrion help employees do?\" -> search_documents (a company/"
    "product name in the question means look it up, never answer from assumed "
    "recognition of that name)\n"
    "Q: \"What does Acme's onboarding process look like?\" -> search_documents "
    "(same pattern with a different, unfamiliar-sounding name)\n"
    "Q: \"What does the engineering team prioritize right now?\" -> search_documents "
    "(a planning/priority question is answered from docs, not a headcount/status lookup)\n"
    "Q: \"How many employees do we have?\" -> query_company_data\n"
    "Q: \"What is the status of Project Atlas?\" -> query_company_data\n"
    "Q: \"What is the capital of France?\" -> no_tool_needed (a universal fact with "
    "no dependency on any company)\n"
    "Q: \"What's 2 + 2?\" -> no_tool_needed"
)


class RoutingFailedError(Exception):
    pass


def route_question(question: str) -> tuple[str, dict]:
    last_err: Exception | None = None

    for attempt in range(1, ROUTING_MAX_ATTEMPTS + 1):
        try:
            decision = groq_client.chat.completions.create(
                model=ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                tools=TOOLS,
                tool_choice="required",
                temperature=0.0,
            )
            tool_calls = decision.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("Model returned no tool_calls despite tool_choice=required")

            call = tool_calls[0]
            tool_name = call.function.name
            args = json.loads(call.function.arguments) if call.function.arguments else {}
            return tool_name, args

        except Exception as err:
            last_err = err
            continue

    raise RoutingFailedError(str(last_err))


def generate_answer(prompt_or_question: str) -> str:
    response = groq_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt_or_question}],
        temperature=0.1,
        max_tokens=1500,
        frequency_penalty=1.0,
    )
    return response.choices[0].message.content


_FINAL_ANSWER_MARKER_RE = re.compile(
    r"[=\*\s]*=={2,}\s*FINAL\s+ANSWER\s*={2,}[=\*\s]*",
    re.IGNORECASE,
)


def extract_final_answer(raw_answer: str, *, question: str = "") -> str:
    match = _FINAL_ANSWER_MARKER_RE.search(raw_answer)
    if not match:
        generation_logger.warning(
            "extract_final_answer: no FINAL ANSWER marker found (question=%r); "
            "returning raw model output unstripped - answer may include "
            "reasoning scratch-work. Raw output: %r",
            question, raw_answer,
        )
        return raw_answer.strip()
    return raw_answer[match.end():].strip()


def generate_answer_with_marker_retry(prompt: str, question: str) -> str:
    for attempt in range(MAX_GENERATION_RETRIES):
        raw = generate_answer(prompt)
        if _FINAL_ANSWER_MARKER_RE.search(raw):
            return extract_final_answer(raw, question=question)
        generation_logger.warning(
            "generate_answer_with_marker_retry: attempt %d/%d missing marker, retrying (question=%r)",
            attempt + 1, MAX_GENERATION_RETRIES, question,
        )
    generation_logger.error(
        "generate_answer_with_marker_retry: all %d attempts missing marker (question=%r)",
        MAX_GENERATION_RETRIES, question,
    )
    return "I ran into an issue generating a clean answer — please try rephrasing your question."


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    chunks_used: int
    tool_called: str


class ShareDocumentRequest(BaseModel):
    email: str
    permission: str = "viewer"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, token_payload: dict = Depends(verify_token)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    user_id = get_or_create_user(email, google_id)

    tool_called = "none"

    try:
        try:
            tool_called, args = route_question(req.question)
        except RoutingFailedError as routing_err:
            tool_called = "routing_failed"
            log_agent_call(
                user_id, req.question, "routing_failed", False,
                reasoning=f"Tool routing failed after {ROUTING_MAX_ATTEMPTS} attempts: {routing_err}",
            )
            answer = (
                "I couldn't reliably determine how to answer that. "
                "Please try rephrasing your question."
            )
            return ChatResponse(answer=answer, chunks_used=0, tool_called=tool_called)

        if tool_called == "no_tool_needed":
            answer = generate_answer(req.question)
            log_agent_call(
                user_id, req.question, "no_tool_needed", True,
                reasoning="Model determined this question needs no company data or documents",
            )
            return ChatResponse(answer=answer, chunks_used=0, tool_called="no_tool_needed")

        if tool_called == "search_documents":
            raw_search_query = args.get("query", req.question)
            search_query = sanitize_router_query(raw_search_query, req.question)
            sub_questions = decompose_question(req.question)

            seen: set[tuple[str, str]] = set()
            chunks: list[RetrievedChunk] = []
            for sub_q in sub_questions:
                sub_chunks = search_chunks_for_user(sub_q, user_id)
                for c in sub_chunks:
                    key = (c.document_id, c.content)
                    if key not in seen:
                        seen.add(key)
                        chunks.append(c)

            if not chunks:
                log_agent_call(
                    user_id, req.question, tool_called, True,
                    reasoning=(
                        f"Searched for '{search_query}' ({len(sub_questions)} "
                        f"sub-question(s)), no chunks within relevance "
                        f"threshold (max_distance={MAX_CHUNK_DISTANCE})"
                    ),
                )
                return ChatResponse(answer=NO_CONTEXT_ANSWER, chunks_used=0, tool_called=tool_called)

            prompt = build_document_prompt(req.question, chunks)
            answer = generate_answer_with_marker_retry(prompt, req.question)
            sources = ", ".join(sorted({c.title for c in chunks}))
            log_agent_call(
                user_id, req.question, tool_called, True,
                reasoning=(
                    f"Used {len(chunks)} relevant document chunk(s) from "
                    f"[{sources}] across {len(sub_questions)} sub-question(s) "
                    f"to answer"
                ),
            )
            return ChatResponse(answer=answer, chunks_used=len(chunks), tool_called=tool_called)

        if tool_called == "query_company_data":
            user_role = get_user_role(user_id)
            record_type = args.get("record_type")
            records = query_internal_records(record_type, user_role)

            if not records:
                answer = f"I don't have access to {record_type} records, or none exist."
                log_agent_call(
                    user_id, req.question, tool_called, True,
                    reasoning=f"Queried '{record_type}' records as role '{user_role}', found 0 accessible",
                )
                return ChatResponse(answer=answer, chunks_used=0, tool_called=tool_called)

            prompt = build_records_prompt(req.question, records)
            answer = generate_answer(prompt)
            log_agent_call(
                user_id, req.question, tool_called, True,
                reasoning=f"Used {record_type} records (role: {user_role}) to answer",
            )
            return ChatResponse(answer=answer, chunks_used=0, tool_called=tool_called)

        raise HTTPException(status_code=500, detail=f"Unknown tool: {tool_called}")

    except HTTPException:
        raise
    except Exception as e:
        log_agent_call(user_id, req.question, tool_called, False, reasoning=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    token_payload: dict = Depends(verify_token),
):
    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    user_id = get_or_create_user(email, google_id)
    title = file.filename

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    existing = None
    chunks: list[str] = []

    try:
        text = extract_text(tmp_path)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        conn = get_conn()
        try:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT d.id, d.content_hash FROM documents d
                JOIN document_permissions dp ON dp.document_id = d.id
                WHERE d.title = %s AND dp.user_id = %s AND d.is_current = true
                """,
                (title, user_id),
            )
            existing = cur.fetchone()

            if existing is not None:
                existing_id, existing_hash = existing
                if existing_hash == content_hash:
                    return {
                        "document_id": str(existing_id),
                        "chunks_stored": 0,
                        "status": "duplicate_skipped",
                        "message": (
                            "A document with this title and identical "
                            "content already exists - no changes made."
                        ),
                    }
                cur.execute(
                    "UPDATE documents SET is_current = false WHERE id = %s",
                    (existing_id,),
                )

            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)

            document_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO documents (id, source, title, content_hash, is_current)
                VALUES (%s, %s, %s, %s, true)
                """,
                (document_id, "upload", title, content_hash),
            )

            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    "INSERT INTO chunks (id, document_id, content, embedding) VALUES (%s, %s, %s, %s)",
                    (str(uuid.uuid4()), document_id, chunk, embedding),
                )

            cur.execute(
                "INSERT INTO document_permissions (document_id, user_id, permission) VALUES (%s, %s, %s)",
                (document_id, user_id, "owner"),
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "document_id": document_id,
            "chunks_stored": len(chunks),
            "status": "superseded_previous_version" if existing is not None else "created",
        }

    finally:
        os.unlink(tmp_path)


@app.get("/internal/{record_type}")
def get_internal_records(record_type: str, token_payload: dict = Depends(verify_token)):
    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    user_id = get_or_create_user(email, google_id)
    user_role = get_user_role(user_id)

    records = query_internal_records(record_type, user_role)
    return {"type": record_type, "count": len(records), "records": records}


@app.get("/logs")
def get_logs(token_payload: dict = Depends(verify_token), limit: int = 20):
    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    user_id = get_or_create_user(email, google_id)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT query, tool_called, reasoning, success, created_at
            FROM agent_logs
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "query": r[0],
            "tool_called": r[1],
            "reasoning": r[2],
            "success": r[3],
            "created_at": r[4].isoformat(),
        }
        for r in rows
    ]


@app.get("/documents")
def list_documents(token_payload: dict = Depends(verify_token)):
    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    user_id = get_or_create_user(email, google_id)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.id, d.title, d.created_at, dp.permission
            FROM documents d
            JOIN document_permissions dp ON dp.document_id = d.id
            WHERE dp.user_id = %s AND d.is_current = true
            ORDER BY d.created_at DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"id": r[0], "title": r[1], "created_at": r[2].isoformat(), "permission": r[3]}
        for r in rows
    ]


@app.post("/documents/{document_id}/share")
def share_document(
    document_id: str,
    req: ShareDocumentRequest,
    token_payload: dict = Depends(verify_token),
):
    if req.permission not in ("viewer", "owner"):
        raise HTTPException(
            status_code=400,
            detail="permission must be 'viewer' or 'owner'",
        )

    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    requester_id = get_or_create_user(email, google_id)

    conn = get_conn()
    try:
        cur = conn.cursor()

        require_document_owner(cur, document_id, requester_id)

        target_email = req.email.strip().lower()
        if target_email == email.strip().lower():
            raise HTTPException(
                status_code=400,
                detail="You already have access to this document.",
            )

        target_user_id = get_or_create_user(target_email)

        cur.execute(
            """
            INSERT INTO document_permissions (document_id, user_id, permission)
            VALUES (%s, %s, %s)
            ON CONFLICT (document_id, user_id)
            DO UPDATE SET permission = EXCLUDED.permission
            """,
            (document_id, target_user_id, req.permission),
        )
        conn.commit()

        return {
            "document_id": document_id,
            "shared_with": target_email,
            "permission": req.permission,
        }
    finally:
        conn.close()


@app.delete("/documents/{document_id}/share/{target_email}")
def revoke_document_access(
    document_id: str,
    target_email: str,
    token_payload: dict = Depends(verify_token),
):
    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    requester_id = get_or_create_user(email, google_id)

    target_email = target_email.strip().lower()
    if target_email == email.strip().lower():
        raise HTTPException(
            status_code=400,
            detail="Owners cannot revoke their own access via this endpoint.",
        )

    conn = get_conn()
    try:
        cur = conn.cursor()

        require_document_owner(cur, document_id, requester_id)

        cur.execute("SELECT id FROM users WHERE email = %s", (target_email,))
        target_row = cur.fetchone()
        if target_row is None:
            raise HTTPException(status_code=404, detail="No user found with that email.")
        target_user_id = str(target_row[0])

        cur.execute(
            "DELETE FROM document_permissions WHERE document_id = %s AND user_id = %s",
            (document_id, target_user_id),
        )
        conn.commit()

        return {"document_id": document_id, "revoked_from": target_email}
    finally:
        conn.close()


@app.get("/documents/{document_id}/shares")
def list_document_shares(
    document_id: str,
    token_payload: dict = Depends(verify_token),
):
    """List every user a document is currently shared with, so the owner
    can see who has access before deciding whether to revoke anyone.

    Only the document's owner can view this list - a viewer calling this
    endpoint gets the same "not found" response as someone with no access
    at all, so this doesn't leak the sharing list to non-owners. NOTE:
    intentionally NOT using require_document_owner here, since that helper
    returns 403 for non-owner-with-access, which would itself leak that
    the document exists. This endpoint must return 404 in both cases.
    """
    email = token_payload.get("email")
    google_id = token_payload.get("sub")
    requester_id = get_or_create_user(email, google_id)

    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT dp.permission FROM document_permissions dp
            JOIN documents d ON d.id = dp.document_id
            WHERE dp.document_id = %s AND dp.user_id = %s AND d.is_current = true
            """,
            (document_id, requester_id),
        )
        row = cur.fetchone()
        if row is None or row[0] != "owner":
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have access to it.",
            )

        cur.execute(
            """
            SELECT u.email, dp.permission
            FROM document_permissions dp
            JOIN users u ON u.id = dp.user_id
            WHERE dp.document_id = %s AND dp.user_id != %s
            ORDER BY u.email
            """,
            (document_id, requester_id),
        )
        shares = [{"email": r[0], "permission": r[1]} for r in cur.fetchall()]

        return {"document_id": document_id, "shares": shares}
    finally:
        conn.close()
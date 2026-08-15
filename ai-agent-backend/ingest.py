import os
import re
import uuid
import logging
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer
import pdfplumber

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Load the embedding model once (downloads automatically on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB, matches frontend limit
CHUNK_SIZE = 800   # target characters per chunk (soft target, see chunk_text)
CHUNK_OVERLAP = 150  # characters of overlap between consecutive chunks

# Inserted after each normalized "Label: value" heading-group sentence
# (see normalize_flattened_tables) to force chunk_text() to start a new
# chunk there, rather than packing multiple departments' sentences into
# one chunk together. This matters because a document with more than 2
# departments (e.g. a 4-department contacts table) previously produced one
# combined chunk where each department's info was a smaller fraction of
# the chunk's total text - this diluted the embedding for queries about
# any single department, causing that department's on-call/lead/tool info
# to rank below the vector-distance/rerank cutoffs even though a 2-
# department document's equivalent chunk (a larger fraction of shorter
# text) surfaced fine for the same query. Splitting each department into
# its own chunk gives it an undiluted embedding regardless of how many
# other departments exist in the same source table.
_CHUNK_BOUNDARY_MARKER = "\n\x00CHUNK_BOUNDARY\x00\n"

_LABEL_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z \-/]{0,30}):\s*(.+)$")

# A pipe- or multi-space-delimited data row, e.g. one line of a flattened
# tenure/benefits table:
#   "0-2               | 12                  | 8"
#   "0-2   12   8"
# Requires at least 3 columns so we don't accidentally match incidental
# text that happens to contain a pipe.
_TABLE_ROW_RE = re.compile(r"^(\S[\w+\-\s]{0,20}?)\s*(?:\||\s{2,})\s*(\d+[\w%]*)\s*(?:\||\s{2,})\s*(\d+[\w%]*)\s*$")

# Matches a plausible column-header row for the above, e.g.
# "Years at Company | Vacation Days/Year | Sick Days/Year"
_TABLE_HEADER_RE = re.compile(r"^(\S.{0,40}?)\s*(?:\||\s{2,})\s*(\S.{0,40}?)\s*(?:\||\s{2,})\s*(\S.{0,40})$")

# Sentence-ish boundary: end punctuation followed by whitespace, or a
# blank line (paragraph break). Used to avoid cutting chunks mid-word or
# mid-header.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+|\n\s*\n")


class IngestError(Exception):
    """Raised when a document can't be usefully ingested (empty, too large, etc)."""


def extract_text(file_path: str) -> str:
    """Extract raw text from a PDF or plain text file.

    For PDFs: uses pdfplumber instead of pypdf. Any table pdfplumber's
    layout detector recognizes (a real grid-drawn table, e.g. a tenure/
    vacation table or a side-by-side department contacts grid) is
    converted directly into clean "label: value" sentences at
    extraction time, using the table's actual row/column structure -
    this avoids relying on normalize_flattened_tables/
    normalize_data_tables below to reconstruct that structure from
    flattened, ambiguous text after the fact.

    Any text NOT inside a detected table (prose, section headings, and
    tables pdfplumber's detector doesn't recognize as a grid - e.g. a
    plain pipe-delimited single-line table with no drawn lines) is
    extracted as plain text, unchanged. normalize_flattened_tables/
    normalize_data_tables still run on that text afterward exactly as
    before, so those functions remain necessary and are not being
    removed - they now simply have less work to do, since real grid
    tables are handled directly here instead.

    Failure handling (this is the part that changed from the initial
    pdfplumber port): pdfplumber's table/text extraction can fail for
    ordinary, expected reasons - not just theoretical edge cases:
      - a malformed/corrupted page object, an unusual font, or a
        pdfminer.six parsing quirk can raise inside find_tables() or
        extract_words() for a SPECIFIC page while every other page in
        the same document is completely fine;
      - rotated pages, multi-column layouts, or nested tables can
        produce bbox coordinates that don't map cleanly back onto
        words, which is handled defensively inside
        _page_text_with_tables_normalized itself rather than raised;
      - the file itself can be encrypted/permission-restricted, or
        just not a valid PDF, in which case pdfplumber.open() fails
        before any page is ever touched.

    A single try/except wrapped around the whole function would hide
    *which* page or *which* mechanism broke, and would abort ingestion
    of an otherwise-fine document because of one bad page. So handling
    is layered instead:
      1. pdfplumber.open() is wrapped on its own - a totally unreadable
         file (encrypted, corrupt, not actually a PDF) raises a clear
         IngestError immediately, with no per-page ambiguity possible.
      2. Each page's table-processing step is wrapped individually - if
         a page raises, that ONE page falls back to plain
         page.extract_text() (or "" if that also fails), a warning is
         logged with the page number, and every other page is
         unaffected.
      3. If every single page fails table processing AND plain text
         extraction, that surfaces via the existing empty-text check in
         store_document() (no extractable text -> IngestError), so
         callers still get a clear failure rather than silently storing
         an empty document.
    """
    if not file_path.lower().endswith(".pdf"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    try:
        pdf = pdfplumber.open(file_path)
    except Exception as e:
        raise IngestError(
            f"Could not open PDF '{file_path}': {e}. "
            "The file may be encrypted, corrupted, or not a valid PDF."
        ) from e

    out_pages = []
    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.find_tables()
            except Exception as e:
                logger.warning(
                    "pdfplumber table detection failed on page %d of '%s' (%s); "
                    "falling back to plain text extraction for this page.",
                    page_num, file_path, e,
                )
                out_pages.append(_safe_plain_text(page, page_num, file_path))
                continue

            if not tables:
                # Scanned/image-only PDFs have no text layer -
                # extract_text() returns None or "" in that case.
                out_pages.append(_safe_plain_text(page, page_num, file_path))
                continue

            try:
                out_pages.append(_page_text_with_tables_normalized(page, tables))
            except Exception as e:
                logger.warning(
                    "pdfplumber table normalization failed on page %d of '%s' (%s); "
                    "falling back to plain text extraction for this page.",
                    page_num, file_path, e,
                )
                out_pages.append(_safe_plain_text(page, page_num, file_path))
    finally:
        pdf.close()

    return "\n".join(out_pages)


def _safe_plain_text(page, page_num: int, file_path: str) -> str:
    """Best-effort plain-text fallback for a single page. Returns "" (not
    a raise) on failure, so one unreadable page never aborts the whole
    document - the overall empty-text check in store_document() is what
    catches the case where EVERY page fails."""
    try:
        return page.extract_text() or ""
    except Exception as e:
        logger.warning(
            "Plain text extraction also failed on page %d of '%s' (%s); "
            "this page will contribute no text.",
            page_num, file_path, e,
        )
        return ""


def _word_in_any_bbox(word, bboxes) -> bool:
    x0, top, x1, bottom = word["x0"], word["top"], word["x1"], word["bottom"]
    for (bx0, btop, bx1, bbottom) in bboxes:
        if x0 >= bx0 - 1 and x1 <= bx1 + 1 and top >= btop - 1 and bottom <= bbottom + 1:
            return True
    return False


def _table_to_sentences(table) -> list[str]:
    """Convert one pdfplumber Table object into normalized sentences.

    Two shapes are handled, matching the two flattened-table shapes
    normalize_flattened_tables/normalize_data_tables already target:
      - A numeric header+rows table (e.g. tenure -> vacation/sick days),
        detected by every header cell being non-empty text.
      - A side-by-side entity grid (e.g. two departments per row, each
        cell holding a "Lead:\\nSlack:\\n..." block), detected by cell
        content containing embedded newlines.
    Returns [] if the table doesn't match either recognizable shape, so
    the caller falls back to that region's raw text instead of guessing
    - the existing regex normalization gets a chance at it downstream.
    """
    rows = table.extract()
    if not rows or len(rows) < 2:
        return []

    header = rows[0]
    sentences = []

    body_has_newlines = any(
        cell and "\n" in cell for row in rows[1:] for cell in row if cell
    )

    if body_has_newlines:
        # Side-by-side department/entity grid: heading row, then a
        # "Label: value\nLabel: value" row, repeating.
        for r_idx in range(0, len(rows) - 1, 2):
            headings = rows[r_idx]
            bodies = rows[r_idx + 1] if r_idx + 1 < len(rows) else []
            for heading, body in zip(headings, bodies):
                if not heading or not body:
                    continue
                field_str = "; ".join(ln.strip() for ln in body.split("\n") if ln.strip())
                sentences.append(f"{heading.strip()}: {field_str}.")
    elif header and all(header):
        # Numeric tenure-style table: header + one data row per line.
        row_label_col, *value_cols = header
        for row in rows[1:]:
            if not row or not row[0]:
                continue
            row_key, *vals = row
            if len(vals) != len(value_cols):
                continue
            parts = "; ".join(f"{col} is {val}" for col, val in zip(value_cols, vals))
            sentences.append(f"For {row_label_col.rstrip('s')} {row_key}: {parts}.")

    return sentences


def _page_text_with_tables_normalized(page, tables) -> str:
    """Return a page's text with every detected table region replaced by
    its normalized sentences, spliced back in at the table's original
    vertical position so reading order (prose -> table -> prose) is
    preserved for downstream chunking.

    Each generated sentence is wrapped with the same hard
    _CHUNK_BOUNDARY_MARKER that normalize_flattened_tables/
    normalize_data_tables use for their own generated sentences (one
    per department/row). Without this, chunk_text's sentence-packer
    would pack multiple rows or departments - or this table plus
    unrelated surrounding prose - into one chunk, diluting each row's
    embedding the same way the original flattened-table bug did before
    any normalization existed.

    Per-table failures (e.g. a single malformed table's bbox or extract()
    raising) are caught individually so one bad table on a page doesn't
    lose the rest of that page's tables or its surrounding prose - that
    table's region simply falls back to raw pipe-joined cell text, same
    as the existing "unrecognized shape" fallback below.
    """
    table_bboxes = [t.bbox for t in tables]
    words = page.extract_words()
    non_table_words = [w for w in words if not _word_in_any_bbox(w, table_bboxes)]

    lines_by_top: dict[int, list] = {}
    for w in non_table_words:
        lines_by_top.setdefault(round(w["top"]), []).append(w)
    positioned_lines = [
        (top, " ".join(w["text"] for w in sorted(line_words, key=lambda w: w["x0"])))
        for top, line_words in lines_by_top.items()
    ]

    for t in tables:
        try:
            sentences = _table_to_sentences(t)
        except Exception as e:
            logger.warning(
                "Failed to normalize a table (bbox=%s) on a page (%s); "
                "falling back to raw cell text for this table.",
                getattr(t, "bbox", "unknown"), e,
            )
            sentences = []

        if not sentences:
            # Unrecognized table shape (or normalization failed above) -
            # fall back to raw text for this region so nothing is
            # silently dropped; existing regex normalization downstream
            # gets a chance at it instead.
            try:
                raw = t.extract()
                fallback_lines = [
                    " | ".join(c for c in row if c) for row in raw if any(row)
                ]
                positioned_lines.extend((t.bbox[1], ln) for ln in fallback_lines)
            except Exception as e:
                logger.warning(
                    "Raw fallback extraction also failed for a table (bbox=%s) (%s); "
                    "this table's content will be omitted.",
                    getattr(t, "bbox", "unknown"), e,
                )
            continue
        marked = [_CHUNK_BOUNDARY_MARKER + s for s in sentences]
        positioned_lines.extend((t.bbox[1], s) for s in marked)

    positioned_lines.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in positioned_lines)


def _is_label_line(line: str) -> bool:
    return bool(_LABEL_LINE_RE.match(line.strip()))


def _find_label_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Find runs of 3+ consecutive "Label: value" lines, e.g.:

        Lead: Marcus Reid
        Slack: #eng-team
        On-call: PagerDuty

    These runs are the classic signature of a PDF table column that got
    flattened to plain text during extraction. Returns (start, end) index
    pairs (end exclusive) for each run found in `lines`.
    """
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_label_line(lines[i]):
            start = i
            while i < n and _is_label_line(lines[i]):
                i += 1
            if i - start >= 2:
                blocks.append((start, i))
        else:
            i += 1
    return blocks


def _split_pairs_into_departments(pairs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Split a flat list of (label, value) pairs into one list per
    department/entity, using the repeating first-label pattern as the
    boundary.

    General fix: find the label that appears as pairs[0]'s label, then
    treat every subsequent occurrence of that SAME label (case-
    insensitive) as a new group boundary, whatever word it happens to be.
    This works for "Lead:", "Manager:", or any other recurring opening
    field, without needing to know the word in advance. A pairs list with
    only one occurrence of that label (i.e. one department/entity, no
    other packed into the same run) still correctly returns a single
    group.
    """
    if not pairs:
        return []
    boundary_label = pairs[0][0].strip().lower()
    groups: list[list[tuple[str, str]]] = []
    for label, value in pairs:
        if label.strip().lower() == boundary_label or not groups:
            groups.append([])
        groups[-1].append((label, value))
    return groups


def normalize_flattened_tables(text: str) -> str:
    """Rewrite runs of "Label: value" lines into an explicit sentence so
    that a downstream LLM doesn't have to infer groupings from ordering.

    Department boundaries within a label-run are found via
    _split_pairs_into_departments() (splitting on "Lead:"), which is
    reliable regardless of department count. Heading NAMES are then
    assigned per group as follows:
      - If the number of heading words found immediately above the block
        equals the number of departments found, each department gets its
        own heading word, in order.
      - Otherwise, the full heading text is reused for every department
        group.
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    blocks = _find_label_blocks(lines)
    if not blocks:
        return text

    out_lines = list(lines)
    # Process blocks in reverse so earlier edits don't shift later indices.
    for start, end in reversed(blocks):
        block_lines = lines[start:end]
        pairs = []
        for ln in block_lines:
            m = _LABEL_LINE_RE.match(ln.strip())
            if m:
                pairs.append((m.group(1).strip(), m.group(2).strip()))

        department_groups = _split_pairs_into_departments(pairs)

        head_idx = start - 1
        heading_lines = []
        while head_idx >= 0:
            candidate = lines[head_idx].strip()
            if candidate == "":
                head_idx -= 1
                continue
            if _is_label_line(candidate):
                break
            if ":" in candidate:
                break  # section title or other colon-bearing line, not a heading
            if len(candidate.split()) <= 4:
                heading_lines.insert(0, candidate)
                head_idx -= 1
                if len(heading_lines) >= 2:
                    break
            else:
                break

        if not heading_lines:
            continue  # no identifiable heading - leave this block as-is

        heading_words: list[str] = []
        for hl in heading_lines:
            heading_words.extend(hl.split())

        if len(heading_words) == len(department_groups):
            groups = list(zip(heading_words, department_groups))
        else:
            full_heading = " ".join(heading_lines)
            groups = [(full_heading, grp) for grp in department_groups]

        sentences = []
        for heading, group_pairs in groups:
            if not group_pairs:
                continue
            field_str = "; ".join(f"{label} is {value}" for label, value in group_pairs)
            sentences.append(f"{heading}: {field_str}.")

        if sentences:
            replace_start = head_idx + 1
            out_lines[replace_start:end] = [
                _CHUNK_BOUNDARY_MARKER + _CHUNK_BOUNDARY_MARKER.join(sentences)
            ]

    return "\n".join(out_lines)


def _find_data_table_blocks(lines: list[str]) -> list[tuple[int, int, tuple[str, str, str]]]:
    """Find a header row matching _TABLE_HEADER_RE followed by 2+ rows
    matching _TABLE_ROW_RE - the signature of a flattened numeric table.
    Tolerates interrupting non-table-row lines in the middle of a
    table's data rows, up to MAX_INTERRUPTION_LINES.
    """
    MAX_INTERRUPTION_LINES = 4
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        header_match = _TABLE_HEADER_RE.match(lines[i].strip())
        if header_match:
            col_labels = tuple(g.strip() for g in header_match.groups())
            j = i + 1
            last_row_end = i
            gap_len = 0
            while j < n:
                if _TABLE_ROW_RE.match(lines[j].strip()):
                    j += 1
                    last_row_end = j
                    gap_len = 0
                elif gap_len < MAX_INTERRUPTION_LINES:
                    gap_len += 1
                    j += 1
                else:
                    break
            end = last_row_end
            if end - (i + 1) >= 2:
                blocks.append((i, end, col_labels))
                i = end
                continue
        i += 1
    return blocks


def normalize_data_tables(text: str) -> str:
    """Rewrite flattened numeric tables (header row + N data rows, columns
    separated by '|' or multiple spaces) into one explicit sentence per
    row, so each row's values are unambiguously tied to their row label
    and column names instead of relying on positional/proximity inference
    at answer time.
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    blocks = _find_data_table_blocks(lines)
    if not blocks:
        return text

    out_lines = list(lines)
    for start, end, col_labels in reversed(blocks):
        col1_label, col2_label, col3_label = col_labels
        sentences = []
        for row_line in lines[start + 1:end]:
            m = _TABLE_ROW_RE.match(row_line.strip())
            if not m:
                continue
            row_key, val2, val3 = (g.strip() for g in m.groups())
            sentences.append(
                f"For {col1_label.rstrip('s')} {row_key}: {col2_label} is {val2}, "
                f"{col3_label} is {val3}."
            )
        if sentences:
            out_lines[start:end] = sentences

    return "\n".join(out_lines)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-ish units on punctuation/paragraph breaks,
    never mid-word. Falls back to the whole string as one unit if no
    boundary is found (e.g. a single long line with no punctuation)."""
    pieces = _SENTENCE_SPLIT_RE.split(text)
    return [p for p in (piece.strip() for piece in pieces) if p]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of roughly chunk_size characters, packed
    from whole sentences/lines so a chunk boundary never falls mid-word
    or mid-header.

    Text is first passed through normalize_flattened_tables() and
    normalize_data_tables() so that multi-column tables flattened during
    PDF extraction (both "Label: value" blocks and header+row numeric
    tables) become unambiguous per-entity sentences instead of relying on
    an LLM to infer groupings from line ordering at answer time.

    Sentences longer than chunk_size (rare) are kept whole rather than
    force-split, since a mid-word cut is worse than a slightly oversized
    chunk.
    """
    text = normalize_flattened_tables(text)
    text = normalize_data_tables(text)

    segments = text.split(_CHUNK_BOUNDARY_MARKER)

    chunks: list[str] = []
    for segment in segments:
        chunks.extend(_chunk_segment(segment, chunk_size, overlap))

    return chunks


def _chunk_segment(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack one boundary-delimited segment's sentences into chunks of
    roughly chunk_size characters. Never called across a hard chunk
    boundary - see chunk_text()."""
    units = _split_into_sentences(text)
    if not units:
        return []

    chunks = []
    current: list[str] = []
    current_len = 0

    i = 0
    while i < len(units):
        unit = units[i]
        added_len = len(unit) + (1 if current else 0)
        if current and current_len + added_len > chunk_size:
            chunks.append(" ".join(current))
            overlap_units: list[str] = []
            overlap_len = 0
            for prev_unit in reversed(current):
                overlap_units.insert(0, prev_unit)
                overlap_len += len(prev_unit) + 1
                if overlap_len >= overlap:
                    break
            current = overlap_units
            current_len = sum(len(u) for u in current) + max(len(current) - 1, 0)
            continue
        current.append(unit)
        current_len += added_len
        i += 1

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def embed_chunks(chunks: list[str]):
    """Convert a list of text chunks into embedding vectors."""
    return model.encode(chunks).tolist()  # model returns NumPy array, convert to python list


def validate_file(file_path: str):
    """Raise IngestError early for files we shouldn't even try to process."""
    size = os.path.getsize(file_path)
    if size == 0:
        raise IngestError(f"File is empty: {file_path}")
    if size > MAX_FILE_SIZE_BYTES:
        raise IngestError(
            f"File is too large ({size} bytes). Max allowed is {MAX_FILE_SIZE_BYTES} bytes."
        )


def store_document(file_path: str, title: str, user_id: str, source: str = "upload") -> str:
    """Full pipeline: extract -> normalize -> chunk -> embed -> store in
    Postgres, and grant the given user access to the resulting document.

    user_id is required - without a document_permissions row, nothing
    ingested here would ever be retrievable via search_chunks_for_user.
    """
    validate_file(file_path)

    text = extract_text(file_path)
    if not text.strip():
        raise IngestError(
            f"No extractable text found in '{title}'. "
            "This usually means the PDF is scanned/image-based and has no "
            "text layer - OCR would be needed to make it searchable - or "
            "every page failed extraction (see logs for per-page warnings)."
        )

    chunks = chunk_text(text)
    if not chunks:
        raise IngestError(f"Text was extracted from '{title}' but produced no usable chunks.")

    embeddings = embed_chunks(chunks)

    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()

        document_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO documents (id, source, title) VALUES (%s, %s, %s)",
            (document_id, source, title),
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

    print(f"Stored '{title}' as {len(chunks)} chunks (document_id={document_id})")
    return document_id


if __name__ == "__main__":
    # Quick manual test - replace with a real file path and a real user_id
    # (a UUID from your users table) before running standalone.
    TEST_FILE_PATH = "test_document.txt"
    TEST_USER_ID = "replace-with-a-real-user-uuid"

    try:
        store_document(TEST_FILE_PATH, "Test Document", TEST_USER_ID)
    except IngestError as e:
        print(f"Ingestion failed: {e}")
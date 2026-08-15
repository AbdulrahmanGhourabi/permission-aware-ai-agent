"""Evaluation set for the permission-aware AI agent.

Each test case defines:
- question: what to ask /chat
- expected_tool: which tool the agent SHOULD choose ("search_documents", "query_company_data", or "no_tool_needed")
- expected_keywords: words/phrases that should appear in a correct answer (case-insensitive)

This suite is built against the CURRENT corpus as of this project's
retrieval/synthesis debugging work: Acme_Corp_Employee_Handbook.pdf,
Borealis_Inc_Employee_Handbook.pdf, sentrion_pdf.pdf, and
Vantable_Systems_Handbook.pdf. Every question/expected-answer pair below
was manually verified against the live app during development - these
are not guesses, they're a regression suite locking in behavior that was
confirmed correct by hand.

If you re-ingest a different corpus, these cases will no longer be
meaningful and need to be rewritten against whatever documents are
actually loaded.
"""

EVAL_CASES = [
    # --- Single-document, single-fact lookups ---
    {
        "question": "How many remote days per week does Sentrion allow?",
        "expected_tool": "search_documents",
        "expected_keywords": ["three", "3"],
    },
    {
        "question": "What is Acme's expense pre-approval threshold?",
        "expected_tool": "search_documents",
        "expected_keywords": ["300", "$300"],
    },
    {
        "question": "How many hours must a lost device be reported within at Borealis?",
        "expected_tool": "search_documents",
        "expected_keywords": ["two", "2 hours", "two hours"],
    },
    {
        "question": "What is the pre-approval threshold for IC5-IC6 at Vantable?",
        "expected_tool": "search_documents",
        "expected_keywords": ["450", "$450"],
    },

    # --- Multi-document disambiguation (the core hard case this project debugged) ---
    {
        "question": "Compare Acme's 0-2 sick-day count against Borealis's 0-1 sick-day count, are they the same number?",
        "expected_tool": "search_documents",
        "expected_keywords": ["6", "8", "not the same", "different"],
    },
    {
        "question": "Which company's on-call tool is PagerDuty for Engineering?",
        "expected_tool": "search_documents",
        "expected_keywords": ["sentrion", "borealis"],
    },
    {
        "question": "Extract the full vacation and sick day table for the company with director-level full-remote approval.",
        "expected_tool": "search_documents",
        "expected_keywords": ["borealis", "15", "20", "25", "30"],
    },
    {
        "question": "How many remote days per week and how many hours of advance notice does Sentrion require?",
        "expected_tool": "search_documents",
        "expected_keywords": ["three", "48"],
    },

    # --- Generalized ingestion (Manager: field pattern, not the original Lead: pattern) ---
    {
        "question": "Which team's escalation tool is PagerDuty at Vantable?",
        "expected_tool": "search_documents",
        "expected_keywords": ["data science", "rohan gupta"],
    },

    # --- Number-collision trap (same number, different meaning, same document) ---
    {
        "question": "Is Vantable's 24-hour incident SLA the same thing as its remote-work notice period?",
        "expected_tool": "search_documents",
        "expected_keywords": ["not the same", "unrelated", "different", "no"],
    },

    # --- Compound/decomposed questions ---
    {
        "question": "List every company whose expense reimbursement app is Expensify, and separately confirm whether Acme's remote-work notice period is longer or shorter than Vantable's.",
        "expected_tool": "search_documents",
        "expected_keywords": ["sentrion", "borealis", "longer"],
    },

    # --- Negative case: field that genuinely isn't in the corpus ---
    {
        "question": "What is Vantable's expense reimbursement app?",
        "expected_tool": "search_documents",
        "expected_keywords": ["don't have access", "not", "no information", "not mentioned", "not explicitly"],
    },

    # --- General knowledge (no tool should be used) ---
    {
        "question": "What is the capital of France?",
        "expected_tool": "no_tool_needed",
        "expected_keywords": ["paris"],
    },
    {
        "question": "What is 2 + 2?",
        "expected_tool": "no_tool_needed",
        "expected_keywords": ["4", "four"],
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "expected_tool": "no_tool_needed",
        "expected_keywords": ["shakespeare"],
    },
]
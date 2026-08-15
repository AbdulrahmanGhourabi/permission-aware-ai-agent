"""
Concurrent multi-user permission isolation test - standalone, no app
code changes required.

This does NOT go through HTTP or JWT auth at all. It:
  1. Creates two fake users directly in your `users` table via SQL.
  2. Calls main.py's own internal functions (get_or_create_user,
     search_chunks_for_user) directly, in threads, simulating two users
     hitting the retrieval layer at the same time.
  3. Checks that User A's retrieved chunks never include content only
     User B has permission to, and vice versa.

Your real main.py, verify_token(), and auth flow are never touched or
modified - this test only exercises the retrieval/permission logic that
sits BEHIND auth, which is the part that actually matters for this test
(concurrency safety of the permission-filtered SQL query itself).

SETUP:
  1. Make sure this file sits in the SAME folder as main.py and
     ingest.py, so `import main` works.
  2. Upload one small test document as each fake user BEFORE running
     this - see the upload_test_document() helper below, which does this
     for you automatically on first run.
  3. Just run: python3 concurrency_test_standalone.py
"""

import concurrent.futures
import sys
import uuid

sys.path.insert(0, ".")  # ensure main.py/ingest.py are importable
import main
import ingest

# --- Configure if you want different content -----------------------------
USER_A_EMAIL = "test-user-a@local-test.invalid"
USER_B_EMAIL = "test-user-b@local-test.invalid"

# Unique marker phrases - each user's test doc contains ONLY their own
# marker, so we can detect if it leaks into the other user's results.
USER_A_MARKER = "XYZZY-ALPHA-SECRET-7412"
USER_B_MARKER = "PLUGH-BRAVO-SECRET-9630"

USER_A_DOC_TEXT = f"""Test Document For User A
This document contains a unique marker: {USER_A_MARKER}
This content should only ever be visible to User A.
"""

USER_B_DOC_TEXT = f"""Test Document For User B
This document contains a unique marker: {USER_B_MARKER}
This content should only ever be visible to User B.
"""

NUM_ROUNDS = 15
# ---------------------------------------------------------------------


def ensure_test_user(email: str) -> str:
    """Get or create a fake user directly via main.py's own function -
    no HTTP, no JWT, exercises the exact same DB logic your real app
    uses."""
    return main.get_or_create_user(email=email, google_id=None)


def upload_test_document(user_id: str, title: str, text: str) -> str:
    """Insert a small test document directly, bypassing the /ingest HTTP
    endpoint but using the exact same underlying ingest.py functions your
    real upload flow uses (chunk_text, embed_chunks), so this is a
    faithful test of the real pipeline."""
    conn = main.get_conn()
    try:
        cur = conn.cursor()

        # Skip re-creating if this exact title already exists for this
        # user (idempotent - safe to re-run this script multiple times).
        cur.execute(
            """
            SELECT d.id FROM documents d
            JOIN document_permissions dp ON dp.document_id = d.id
            WHERE d.title = %s AND dp.user_id = %s
            """,
            (title, user_id),
        )
        existing = cur.fetchone()
        if existing:
            return str(existing[0])

        chunks = ingest.chunk_text(text)
        embeddings = ingest.embed_chunks(chunks)

        document_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO documents (id, source, title) VALUES (%s, %s, %s)",
            (document_id, "concurrency_test", title),
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
        return document_id
    finally:
        conn.close()


def search_as_user(user_id: str, question: str) -> list[str]:
    """Call the real search_chunks_for_user() directly - this is the
    exact function your live /chat endpoint calls, exercised here without
    any HTTP/auth layer in between."""
    chunks = main.search_chunks_for_user(question, user_id)
    return [c.content for c in chunks]


def run_round(round_num: int, user_a_id: str, user_b_id: str) -> list[str]:
    """Fire both users' retrieval calls genuinely concurrently via a
    shared thread pool, then check for cross-contamination."""
    problems = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(search_as_user, user_a_id, "test document marker")
        future_b = executor.submit(search_as_user, user_b_id, "test document marker")
        chunks_a = future_a.result()
        chunks_b = future_b.result()

    text_a = " ".join(chunks_a)
    text_b = " ".join(chunks_b)

    if USER_B_MARKER in text_a:
        problems.append(
            f"[round {round_num}] LEAK: User A's retrieved chunks contain "
            f"User B's marker. Chunks: {chunks_a!r}"
        )
    if USER_A_MARKER in text_b:
        problems.append(
            f"[round {round_num}] LEAK: User B's retrieved chunks contain "
            f"User A's marker. Chunks: {chunks_b!r}"
        )

    if USER_A_MARKER not in text_a:
        problems.append(
            f"[round {round_num}] WARNING (not a leak): User A didn't "
            f"retrieve their own marker - check retrieval quality, not a "
            f"security issue by itself."
        )
    if USER_B_MARKER not in text_b:
        problems.append(
            f"[round {round_num}] WARNING (not a leak): User B didn't "
            f"retrieve their own marker - check retrieval quality, not a "
            f"security issue by itself."
        )

    return problems


def main_test():
    print("Setting up test users and documents...")
    user_a_id = ensure_test_user(USER_A_EMAIL)
    user_b_id = ensure_test_user(USER_B_EMAIL)
    print(f"  User A id: {user_a_id}")
    print(f"  User B id: {user_b_id}")

    upload_test_document(user_a_id, "concurrency_test_doc_A.txt", USER_A_DOC_TEXT)
    upload_test_document(user_b_id, "concurrency_test_doc_B.txt", USER_B_DOC_TEXT)
    print("Test documents ready.\n")

    print(f"Running {NUM_ROUNDS} rounds of concurrent User A / User B retrieval...\n")

    total_leaks = 0
    total_warnings = 0

    for i in range(1, NUM_ROUNDS + 1):
        problems = run_round(i, user_a_id, user_b_id)
        if problems:
            for p in problems:
                print(p)
                if "LEAK" in p:
                    total_leaks += 1
                else:
                    total_warnings += 1
        else:
            print(f"[round {i}] OK - no leakage detected")

    print(f"\n{'='*60}")
    print(f"Rounds run: {NUM_ROUNDS}")
    print(f"LEAKS detected: {total_leaks}")
    print(f"Warnings (non-security): {total_warnings}")
    if total_leaks == 0:
        print("PASS: no cross-user permission leakage detected under concurrency.")
    else:
        print("FAIL: cross-user leakage detected - see LEAK lines above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main_test()
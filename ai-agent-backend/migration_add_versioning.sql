    -- Migration: add content-hash-based versioning to documents.
    --
    -- Required before the updated /ingest endpoint in main.py will work -
    -- it queries and writes to content_hash and is_current, which don't
    -- exist in the original schema.
    --
    -- Run this ONCE against your database before restarting the app with
    -- the updated main.py.

    ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS content_hash TEXT,
        ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT true;

    -- Backfill: every existing document row is treated as the current
    -- version (content_hash left NULL for pre-existing rows - they'll only
    -- get a real hash once re-uploaded through the new /ingest logic; NULL
    -- content_hash simply means "no dedup check possible yet" for that row,
    -- which is safe - it just means the first re-upload of that title won't
    -- be recognized as a true duplicate, but will still correctly supersede
    -- it as an edit).
    UPDATE documents SET is_current = true WHERE is_current IS NULL;

    -- Speeds up the is_current filter now present in every retrieval query
    -- (search_chunks_for_user, list_documents) and the title+user lookup
    -- done on every /ingest call.
    CREATE INDEX IF NOT EXISTS idx_documents_title_current
        ON documents (title, is_current);
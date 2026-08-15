import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")


conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Enable pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Users table
#primary key don't allow null,or duplicates,and unique.Even if UUID makes unique values,duplicates might happen(primary key is database-enforced rule).
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL,
    google_id TEXT
);
""")

# Documents table
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    external_id TEXT,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);
""")

# Chunks table (RAG)
cur.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);
""")

# Document permissions (ACLs)
#1st query below,document_id column that references to existing id in documents table.
cur.execute("""
CREATE TABLE IF NOT EXISTS document_permissions (
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE, 
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permission TEXT NOT NULL,
    PRIMARY KEY (document_id, user_id)
);
""")

# Mock internal records (structured data)
cur.execute("""
CREATE TABLE IF NOT EXISTS internal_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL,
    data JSONB NOT NULL,
    allowed_roles TEXT[] NOT NULL
);
""")

# Agent logs (observability)
cur.execute("""
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    query TEXT,
    tool_called TEXT,
    reasoning TEXT,
    success BOOLEAN,
    created_at TIMESTAMP DEFAULT now()
);
""")

conn.commit()
cur.close()
conn.close()

print("All tables created successfully.")
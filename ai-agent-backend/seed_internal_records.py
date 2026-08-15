"""Seed the internal_records table with structured company data (employees,
projects) for testing the query_company_data tool - the structured-lookup
path distinct from search_documents (which reads PDF chunks).

Run once: python seed_internal_records.py

This data is intentionally tied to the same fictional companies whose
handbooks are used for search_documents testing (Acme, Borealis,
Sentrion, Vantable) rather than generic placeholder names, so a reviewer
can recognize the same company context across both tool paths - and so
role/permission testing (allowed_roles below) has a believable reason to
exist: not every employee/project record should be visible to every
role, which is the actual point of the allowed_roles column.
"""
import os
import uuid
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

EMPLOYEES = [
    {"name": "Marcus Reid", "title": "Engineering Manager", "department": "Engineering", "company": "Sentrion"},
    {"name": "Elena Cruz", "title": "Finance Lead", "department": "Finance", "company": "Sentrion"},
    {"name": "Priya Nair", "title": "Sales Lead", "department": "Sales", "company": "Sentrion"},
    {"name": "Tom Okafor", "title": "HR Business Partner", "department": "HR", "company": "Sentrion"},
    {"name": "Sana Malik", "title": "Product Lead", "department": "Product", "company": "Borealis Inc"},
    {"name": "Daniela Ferreira", "title": "Engineering Manager", "department": "Engineering", "company": "Acme Corp"},
    {"name": "Jordan Kim", "title": "Product Lead", "department": "Product", "company": "Acme Corp"},
    {"name": "Rohan Gupta", "title": "Data Science Manager", "department": "Data Science", "company": "Vantable Systems"},
    {"name": "Alicia Chen", "title": "Platform Engineering Manager", "department": "Platform Engineering", "company": "Vantable Systems"},
]

PROJECTS = [
    {"name": "Project Beacon", "department": "Engineering", "company": "Sentrion", "status": "in_progress"},
    {"name": "Q3 Expansion", "department": "Sales", "company": "Sentrion", "status": "in_progress"},
    {"name": "Benefits Overhaul", "department": "HR", "company": "Sentrion", "status": "completed"},
    {"name": "Roadmap Refresh", "department": "Product", "company": "Borealis Inc", "status": "in_progress"},
    {"name": "Budget Review 2026", "department": "Finance", "company": "Sentrion", "status": "in_progress"},
    {"name": "Retrieval Accuracy Initiative", "department": "Data Science", "company": "Vantable Systems", "status": "in_progress"},
]

# allowed_roles gates who can query_company_data for this record type -
# a "member" role can see employee directory info, but project status
# (which may include unannounced/sensitive initiatives) is restricted to
# "manager" and "admin" roles. This mirrors the same principle
# document_permissions enforces for documents, applied to structured
# data instead.
EMPLOYEE_ALLOWED_ROLES = ["member", "manager", "admin"]
PROJECT_ALLOWED_ROLES = ["manager", "admin"]


def seed():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()

        # Idempotent: clear any previous seed data for these two types
        # before re-inserting, so re-running this script doesn't produce
        # duplicate rows the way repeated /ingest calls once did for
        # documents.
        cur.execute("DELETE FROM internal_records WHERE type IN ('employee', 'project')")

        for emp in EMPLOYEES:
            cur.execute(
                "INSERT INTO internal_records (id, type, data, allowed_roles) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "employee", Json(emp), EMPLOYEE_ALLOWED_ROLES),
            )

        for proj in PROJECTS:
            cur.execute(
                "INSERT INTO internal_records (id, type, data, allowed_roles) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "project", Json(proj), PROJECT_ALLOWED_ROLES),
            )

        conn.commit()
        print(f"Seeded {len(EMPLOYEES)} employee records and {len(PROJECTS)} project records.")
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
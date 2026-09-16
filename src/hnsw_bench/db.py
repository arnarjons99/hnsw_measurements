"""Postgres connection helper. Credentials match docker-compose.yml."""

import psycopg
from pgvector.psycopg import register_vector

DSN = "host=localhost port=5432 dbname=hnsw user=hnsw password=hnsw"


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn

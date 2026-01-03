import sqlite3
from contextlib import contextmanager

DB_PATH = "knowledge.db"

def get_connection():
    """
    Create and return a SQLite DB connection
    """
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False  # FastAPI ke liye important
    )
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """
    Context manager for safe DB usage
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

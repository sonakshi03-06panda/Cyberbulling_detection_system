import os
import sqlite3
import json
from datetime import datetime


def get_db_path() -> str:
    return os.environ.get("DATABASE_PATH", "cyberguard.db")


def _get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the analyses table if it doesn't exist."""
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_url   TEXT    NOT NULL,
                video_id    TEXT    NOT NULL,
                video_title TEXT,
                analyzed_at TEXT    NOT NULL,
                total_comments INTEGER,
                toxic_percent  REAL,
                results_json   TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_analysis(
    video_url: str,
    video_id: str,
    video_title: str,
    total_comments: int,
    toxic_percent: float,
    results_json: str,
) -> int:
    """Insert a new analysis record. Returns the new row ID."""
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO analyses
                (video_url, video_id, video_title, analyzed_at, total_comments, toxic_percent, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_url,
                video_id,
                video_title,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                total_comments,
                toxic_percent,
                results_json,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_all_analyses() -> list:
    """Return all analyses ordered by most recent first."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, video_url, video_id, video_title, analyzed_at, total_comments, toxic_percent "
            "FROM analyses ORDER BY analyzed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_analysis_by_id(analysis_id: int) -> dict | None:
    """Return a single analysis row as a dict, or None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"DB initialized at: {get_db_path()}")

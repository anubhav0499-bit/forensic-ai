"""
Conversation Memory — Session-based history with SQLite persistence.
Tracks multi-turn query-answer pairs per (session_id, company_name, agent_id)
and provides a formatted history block for LLM prompt injection.
"""

from __future__ import annotations
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from loguru import logger
from config import DATA_DIR


class ConversationMemory:
    """
    SQLite-backed memory for multi-turn forensic analysis sessions.

    Schema (conversations table):
      id          — auto PK
      session_id  — identifies one analysis run (company + date by default)
      company     — company being investigated
      agent_id    — which agent produced this entry (-1 = user query)
      role        — 'user' | 'assistant'
      content     — the actual query or response text
      metadata    — JSON blob (risk_score, findings_count, etc.)
      created_at  — UTC timestamp

    History context is capped per turn (MAX_CONTENT_CHARS) and per session
    (MAX_HISTORY_TURNS) so it doesn't bloat the LLM context window.
    """

    DB_PATH = DATA_DIR / "conversation_memory.db"
    MAX_HISTORY_TURNS = 5     # user+assistant pairs
    MAX_CONTENT_CHARS = 800   # per turn in the context block

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or self.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Initialisation ────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    company     TEXT    NOT NULL,
                    agent_id    INTEGER NOT NULL DEFAULT -1,
                    role        TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    metadata    TEXT    NOT NULL DEFAULT '{}',
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversations (session_id, company, id)
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ─────────────────────────────────────────────────────────

    def add_turn(
        self,
        session_id: str,
        company_name: str,
        role: str,
        content: str,
        agent_id: int = -1,
        metadata: dict | None = None,
    ) -> None:
        """Persist a single query (role='user') or response (role='assistant')."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations "
                "(session_id, company, agent_id, role, content, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id, company_name, agent_id,
                    role, content, json.dumps(metadata or {}),
                ),
            )

    # ── Read ──────────────────────────────────────────────────────────

    def get_history(
        self,
        session_id: str,
        company_name: str,
        max_turns: int | None = None,
    ) -> list[dict]:
        """
        Return the most recent turns, oldest-first.
        max_turns: number of user+assistant *pairs* to return.
        """
        limit = (max_turns or self.MAX_HISTORY_TURNS) * 2
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, agent_id, created_at "
                "FROM conversations "
                "WHERE session_id=? AND company=? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, company_name, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def build_history_context(
        self,
        session_id: str,
        company_name: str,
    ) -> str:
        """
        Format session history as a compact context block for LLM injection.
        Empty string returned when no history exists.
        """
        turns = self.get_history(session_id, company_name)
        if not turns:
            return ""

        lines = [f"PRIOR ANALYSIS HISTORY — {company_name} (session {session_id[:8]}…):"]
        for turn in turns:
            label = "ANALYST QUERY" if turn["role"] == "user" else "PRIOR FINDING"
            body = turn["content"]
            if len(body) > self.MAX_CONTENT_CHARS:
                body = body[: self.MAX_CONTENT_CHARS] + " …"
            lines.append(f"[{label}] {body}")

        return "\n".join(lines)

    # ── Housekeeping ──────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        """Delete all turns for a session (e.g., after report is generated)."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id=?", (session_id,)
            )
        logger.info(f"Conversation memory cleared for session {session_id}")

    def session_turn_count(self, session_id: str, company_name: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id=? AND company=?",
                (session_id, company_name),
            ).fetchone()
        return row[0] if row else 0

    # ── Session ID helpers ────────────────────────────────────────────

    @staticmethod
    def make_session_id(company_name: str) -> str:
        """One session per company per calendar day (UTC)."""
        date_str = datetime.utcnow().strftime("%Y%m%d")
        raw = f"{company_name.lower().strip()}_{date_str}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def make_run_session_id(company_name: str) -> str:
        """Unique session per invocation (for parallel runs)."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        raw = f"{company_name.lower().strip()}_{ts}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

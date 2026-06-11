"""
SQLite Handler - Transactional storage for all investigation data
"""

from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager
from loguru import logger

from config import DB_CONFIG
from .schema import SCHEMA_SQL


class SQLiteHandler:
    """Thread-safe SQLite handler with WAL mode for concurrent reads."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_CONFIG.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info(f"SQLite initialized: {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"SQLite error: {e}")
            raise
        finally:
            conn.close()

    # ─── Company Operations ─────────────────────────────────
    def upsert_company(self, data: dict) -> int:
        with self._conn() as conn:
            from utils.helpers import normalize_company_name
            data["normalized_name"] = normalize_company_name(data.get("name", ""))
            existing = conn.execute(
                "SELECT id FROM companies WHERE normalized_name = ? OR ticker = ?",
                (data["normalized_name"], data.get("ticker", "")),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE companies SET market_cap=?, sector=?, updated_at=datetime('now') WHERE id=?",
                    (data.get("market_cap"), data.get("sector"), existing["id"]),
                )
                return existing["id"]
            cursor = conn.execute(
                """INSERT INTO companies (name, normalized_name, ticker, isin, exchange, sector, industry, country, currency)
                   VALUES (:name, :normalized_name, :ticker, :isin, :exchange, :sector, :industry, :country, :currency)""",
                data,
            )
            return cursor.lastrowid

    def get_company_id(self, name: str = None, ticker: str = None) -> Optional[int]:
        with self._conn() as conn:
            if ticker:
                row = conn.execute(
                    "SELECT id FROM companies WHERE ticker = ?", (ticker.upper(),)
                ).fetchone()
                if row:
                    return row["id"]
            if name:
                from utils.helpers import normalize_company_name
                norm = normalize_company_name(name)
                row = conn.execute(
                    "SELECT id FROM companies WHERE normalized_name = ?", (norm,)
                ).fetchone()
                if row:
                    return row["id"]
        return None

    # ─── Financial Data ──────────────────────────────────────
    def upsert_financial_data(self, company_id: int, fiscal_year: str, data: dict) -> None:
        data["company_id"] = company_id
        data["fiscal_year"] = fiscal_year
        cols = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        updates = ", ".join(f"{k}=:{k}" for k in data.keys() if k not in ("company_id", "fiscal_year"))
        with self._conn() as conn:
            conn.execute(
                f"""INSERT INTO financial_data ({cols}) VALUES ({placeholders})
                    ON CONFLICT(company_id, fiscal_year) DO UPDATE SET {updates}""",
                data,
            )

    def get_financial_history(self, company_id: int, years: int = 5) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM financial_data WHERE company_id = ?
                   ORDER BY fiscal_year DESC LIMIT ?""",
                (company_id, years),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Documents ──────────────────────────────────────────
    def register_document(self, company_id: int, doc_data: dict) -> int:
        doc_data["company_id"] = company_id
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO documents
                   (company_id, document_type, fiscal_year, quarter, source_url, local_path, file_size_bytes, checksum)
                   VALUES (:company_id, :document_type, :fiscal_year, :quarter, :source_url, :local_path, :file_size_bytes, :checksum)""",
                doc_data,
            )
            return cursor.lastrowid

    def mark_document_parsed(self, doc_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE documents SET parsed=1, parse_date=datetime('now') WHERE id=?",
                (doc_id,),
            )

    def get_documents(self, company_id: int, doc_type: str = None) -> list[dict]:
        with self._conn() as conn:
            if doc_type:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE company_id=? AND document_type=? ORDER BY fiscal_year DESC",
                    (company_id, doc_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE company_id=? ORDER BY fiscal_year DESC",
                    (company_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ─── Forensic Scores ────────────────────────────────────
    def save_forensic_scores(self, company_id: int, fiscal_year: str, scores: dict) -> None:
        scores["company_id"] = company_id
        scores["fiscal_year"] = fiscal_year
        cols = ", ".join(scores.keys())
        placeholders = ", ".join(f":{k}" for k in scores.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO forensic_scores ({cols}) VALUES ({placeholders})",
                scores,
            )

    def get_forensic_scores(self, company_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM forensic_scores WHERE company_id=? ORDER BY fiscal_year DESC",
                (company_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Findings ───────────────────────────────────────────
    def save_finding(self, company_id: int, finding: dict) -> None:
        finding["company_id"] = company_id
        cols = ", ".join(finding.keys())
        placeholders = ", ".join(f":{k}" for k in finding.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO agent_findings ({cols}) VALUES ({placeholders})",
                finding,
            )

    def save_red_flag(self, company_id: int, flag: dict) -> None:
        flag["company_id"] = company_id
        cols = ", ".join(flag.keys())
        placeholders = ", ".join(f":{k}" for k in flag.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO red_flags ({cols}) VALUES ({placeholders})",
                flag,
            )

    def save_green_flag(self, company_id: int, flag: dict) -> None:
        flag["company_id"] = company_id
        cols = ", ".join(flag.keys())
        placeholders = ", ".join(f":{k}" for k in flag.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO green_flags ({cols}) VALUES ({placeholders})",
                flag,
            )

    def get_red_flags(self, company_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM red_flags WHERE company_id=? ORDER BY severity DESC",
                (company_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_findings(self, company_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_findings WHERE company_id=? ORDER BY created_at DESC",
                (company_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Related Parties ────────────────────────────────────
    def save_related_party(self, company_id: int, data: dict) -> None:
        data["company_id"] = company_id
        cols = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO related_party_transactions ({cols}) VALUES ({placeholders})",
                data,
            )

    # ─── Audit History ──────────────────────────────────────
    def save_auditor_record(self, company_id: int, data: dict) -> None:
        data["company_id"] = company_id
        cols = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO auditor_history ({cols}) VALUES ({placeholders})",
                data,
            )

    # ─── Shareholding ───────────────────────────────────────
    def save_shareholding(self, company_id: int, data: dict) -> None:
        data["company_id"] = company_id
        cols = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO shareholding_data ({cols}) VALUES ({placeholders})",
                data,
            )

    # ─── Investigation Session ──────────────────────────────
    def start_session(self, company_id: int) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO investigation_sessions (company_id) VALUES (?)",
                (company_id,),
            )
            return cursor.lastrowid

    def close_session(self, session_id: int, risk_score: float, report_path: str, summary: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE investigation_sessions SET session_end=datetime('now'), status='COMPLETE',
                   overall_risk_score=?, report_path=?, summary=? WHERE id=?""",
                (risk_score, report_path, summary, session_id),
            )

    def execute_query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

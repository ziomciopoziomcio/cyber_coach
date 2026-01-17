"""
components.database
SQLite-backed Database helper for storing exercise session metrics.

Table: sessions
- id INTEGER PRIMARY KEY AUTOINCREMENT
- exercise_name TEXT (optional)
- timestamp TEXT (ISO 8601)
- total_reps INTEGER
- complete_reps INTEGER
- incomplete_reps INTEGER
- avg_rom REAL
- metrics_json TEXT (raw JSON dump of the full metrics dict)

Usage:
    db = Database()  # opens ./components/database.sqlite3
    row_id = db.insert_metrics(metrics_dict, exercise_name='squat')

The module also contains a small performance test when run as a script.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import time
import random
import math

# Public API
__all__ = ["Database"]


def _ensure_dir_exists(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class Database:
    """Simple SQLite-backed database for storing exercise session metrics.

    Each inserted record stores some indexed columns (counts and avg_rom)
    and the full metrics dict as JSON for flexibility.

    The DB file defaults to components/database.sqlite3 next to this file.
    """

    def __init__(self, db_path: Optional[str] = None, timeout: float = 5.0) -> None:
        if db_path is None:
            base = os.path.dirname(__file__)
            db_path = os.path.join(base, "database.sqlite3")

        _ensure_dir_exists(db_path)
        self.db_path = os.path.abspath(db_path)
        self.timeout = timeout
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        # allow access from multiple threads in simple scenarios
        self.conn = sqlite3.connect(self.db_path, timeout=self.timeout, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exercise_name TEXT,
                timestamp TEXT NOT NULL,
                total_reps INTEGER,
                complete_reps INTEGER,
                incomplete_reps INTEGER,
                avg_rom REAL,
                metrics_json TEXT
            )
            """
        )
        # simple index for common queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_exercise ON sessions(exercise_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp)")
        self.conn.commit()

    def insert_metrics(self, metrics: Dict[str, Any], exercise_name: Optional[str] = None,
                       timestamp: Optional[datetime] = None) -> int:
        """Insert a metrics dict into the DB.

        metrics is expected to contain at least these keys (but the function will tolerate missing values):
            - total_reps
            - complete_reps
            - incomplete_reps
            - avg_rom

        The full dict is also stored as JSON in `metrics_json`.
        Returns the row id of the inserted record.
        """
        if timestamp is None:
            # use timezone-aware UTC timestamp
            timestamp = datetime.now(timezone.utc)

        # normalize common fields
        def _to_int(v: Any) -> Optional[int]:
            try:
                if v is None:
                    return None
                return int(v)
            except Exception:
                return None

        def _to_float(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        total_reps = _to_int(metrics.get("total_reps") if metrics is not None else None)
        complete_reps = _to_int(metrics.get("complete_reps"))
        incomplete_reps = _to_int(metrics.get("incomplete_reps"))
        avg_rom = _to_float(metrics.get("avg_rom"))

        # sanitize avg_rom (avoid NaN/inf stored in DB)
        if not (avg_rom is None or math.isfinite(avg_rom)):
            avg_rom = None

        metrics_json = json.dumps(metrics, default=str, ensure_ascii=False)

        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO sessions (exercise_name, timestamp, total_reps, complete_reps, incomplete_reps, avg_rom, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exercise_name,
                timestamp.isoformat(sep=" "),
                total_reps,
                complete_reps,
                incomplete_reps,
                avg_rom,
                metrics_json,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def fetch_by_id(self, row_id: int) -> Optional[Dict[str, Any]]:
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE id = ?", (row_id,))
        row = cur.fetchone()
        if not row:
            return None
        out = dict(row)
        try:
            out["metrics"] = json.loads(out.get("metrics_json") or "null")
        except Exception:
            out["metrics"] = None
        return out

    def fetch_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["metrics"] = json.loads(d.get("metrics_json") or "null")
            except Exception:
                d["metrics"] = None
            result.append(d)
        return result


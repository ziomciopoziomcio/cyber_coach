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


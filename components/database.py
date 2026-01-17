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



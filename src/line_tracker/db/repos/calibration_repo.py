"""Repository for `calibration_thresholds` table."""

from __future__ import annotations

import sqlite3


class CalibrationRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def save(self, key: str, json_str: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO calibration_thresholds
               (key, json, created_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (key, json_str),
        )

    def load(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT json FROM calibration_thresholds WHERE key = ?",
            (key,),
        ).fetchone()
        return row["json"] if row else None

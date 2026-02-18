"""Repository for ``published_slates`` table."""

from __future__ import annotations

import json
import sqlite3


class SlatesRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get_by_unique(
        self,
        slate_date: str,
        sport: str,
        mode: str,
        slate_hash: str,
    ) -> int | None:
        """Return the slate id if a matching row exists, else None."""
        row = self._conn.execute(
            """SELECT id FROM published_slates
               WHERE slate_date = ? AND sport = ? AND mode = ? AND slate_hash = ?""",
            (slate_date, sport, mode, slate_hash),
        ).fetchone()
        return row["id"] if row else None

    def insert(
        self,
        slate_date: str,
        sport: str,
        mode: str,
        thresholds: dict,
        slate_hash: str,
        engine_config: dict,
        code_version: str,
    ) -> int:
        """Insert a new published slate row and return its id."""
        cursor = self._conn.execute(
            """INSERT INTO published_slates
               (slate_date, sport, mode, thresholds, slate_hash,
                engine_config, code_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                slate_date,
                sport,
                mode,
                json.dumps(thresholds, separators=(",", ":")),
                slate_hash,
                json.dumps(engine_config, separators=(",", ":")),
                code_version,
            ),
        )
        return cursor.lastrowid

    def get_latest_for_day(
        self,
        slate_date: str,
        sport: str,
        mode: str,
    ) -> tuple[int, str, str] | None:
        """Return ``(id, created_at, slate_hash)`` for the most recent slate
        matching the given (slate_date, sport, mode), or ``None``."""
        row = self._conn.execute(
            """SELECT id, created_at, slate_hash FROM published_slates
               WHERE slate_date = ? AND sport = ? AND mode = ?
               ORDER BY created_at DESC
               LIMIT 1""",
            (slate_date, sport, mode),
        ).fetchone()
        if row is None:
            return None
        return (row["id"], row["created_at"], row["slate_hash"])

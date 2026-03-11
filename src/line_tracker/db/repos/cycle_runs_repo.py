"""Repository for ``cycle_runs`` table."""

from __future__ import annotations

import json
import sqlite3


class CycleRunsRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_cycle_run(self, result: dict) -> int:
        """Persist a cycle run result dict. Returns the new row id."""
        success = 1 if not result.get("errors") else 0
        cursor = self._conn.execute(
            """INSERT INTO cycle_runs
               (started_at, finished_at, sport, success,
                lines_fetched, events_processed, picks_generated,
                snapshots_written, clv_updates_attempted, clv_updates_completed,
                warnings_json, errors_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.get("started_at"),
                result.get("finished_at"),
                result.get("sport"),
                success,
                result.get("lines_fetched", 0),
                result.get("events_processed", 0),
                result.get("picks_generated", 0),
                result.get("snapshots_written", 0),
                result.get("clv_updates_attempted", 0),
                result.get("clv_updates_completed", 0),
                json.dumps(result.get("warnings", []))
                if result.get("warnings")
                else None,
                json.dumps(result.get("errors", []))
                if result.get("errors")
                else None,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_recent_cycles(self, limit: int = 20) -> list[dict]:
        """Return the most recent cycle runs, newest first."""
        rows = self._conn.execute(
            """SELECT id, started_at, finished_at, sport, success,
                      lines_fetched, events_processed, picks_generated,
                      snapshots_written, clv_updates_attempted,
                      clv_updates_completed, warnings_json, errors_json
               FROM cycle_runs
               ORDER BY started_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

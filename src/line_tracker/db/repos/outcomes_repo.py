"""Repository for the ``outcomes`` table."""

from __future__ import annotations

import sqlite3


class OutcomesRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def import_from_rows(self, rows: list[dict]) -> int:
        """Bulk upsert outcome rows. Returns count of rows inserted/updated."""
        count = 0
        for row in rows:
            cursor = self._conn.execute(
                """INSERT INTO outcomes
                   (event_id, market, selection, result, settled_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(event_id, market, selection)
                   DO UPDATE SET result = excluded.result,
                                 settled_at = excluded.settled_at""",
                (
                    row["event_id"],
                    row["market"],
                    row["selection"],
                    row["result"],
                    row.get("settled_at"),
                ),
            )
            if cursor.rowcount > 0:
                count += 1
        return count

    def get_for_event(
        self, event_id: str, market: str, selection: str,
    ) -> dict | None:
        """Look up a single outcome."""
        row = self._conn.execute(
            """SELECT * FROM outcomes
               WHERE event_id = ? AND market = ? AND selection = ?""",
            (event_id, market, selection),
        ).fetchone()
        return dict(row) if row else None

    def get_all(self) -> list[dict]:
        """Return all outcome rows."""
        rows = self._conn.execute(
            "SELECT * FROM outcomes ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

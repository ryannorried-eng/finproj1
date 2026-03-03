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
            line_value = row.get("line_value")
            if line_value is not None:
                line_value = float(line_value)

            # Manual upsert: partial unique indexes require explicit NULL handling.
            if line_value is None:
                existing = self._conn.execute(
                    """SELECT outcome_id FROM outcomes
                       WHERE event_id = ? AND market = ? AND selection = ?
                       AND line_value IS NULL""",
                    (row["event_id"], row["market"], row["selection"]),
                ).fetchone()
            else:
                existing = self._conn.execute(
                    """SELECT outcome_id FROM outcomes
                       WHERE event_id = ? AND market = ? AND selection = ?
                       AND line_value = ?""",
                    (row["event_id"], row["market"], row["selection"], line_value),
                ).fetchone()

            if existing:
                cursor = self._conn.execute(
                    """UPDATE outcomes SET result = ?, settled_at = ?
                       WHERE outcome_id = ?""",
                    (row["result"], row.get("settled_at"), existing["outcome_id"]),
                )
            else:
                cursor = self._conn.execute(
                    """INSERT INTO outcomes
                       (event_id, market, selection, line_value, result, settled_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        row["event_id"],
                        row["market"],
                        row["selection"],
                        line_value,
                        row["result"],
                        row.get("settled_at"),
                    ),
                )
            if cursor.rowcount > 0:
                count += 1
        return count

    def get_for_event(
        self,
        event_id: str,
        market: str,
        selection: str,
        line_value: float | None = None,
    ) -> dict | None:
        """Look up a single outcome."""
        if line_value is None:
            row = self._conn.execute(
                """SELECT * FROM outcomes
                   WHERE event_id = ? AND market = ? AND selection = ?
                   AND line_value IS NULL""",
                (event_id, market, selection),
            ).fetchone()
        else:
            row = self._conn.execute(
                """SELECT * FROM outcomes
                   WHERE event_id = ? AND market = ? AND selection = ?
                   AND line_value = ?""",
                (event_id, market, selection, line_value),
            ).fetchone()
        return dict(row) if row else None

    def get_all(self) -> list[dict]:
        """Return all outcome rows."""
        rows = self._conn.execute(
            "SELECT * FROM outcomes ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

"""Repository for `lines` table queries/writes."""

from __future__ import annotations

import sqlite3


class LinesRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_many(self, rows: list[tuple]) -> int:
        cursor = self._conn.executemany(
            """INSERT INTO lines
               (sportsbook, sport, event, bet_type, home_team, away_team,
                home_value, away_value, home_price, away_price, timestamp,
                commence_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return cursor.rowcount

    def get_lines(
        self,
        event: str | None = None,
        bet_type: str | None = None,
        sportsbook: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM lines WHERE 1=1"
        params: list = []
        if event:
            query += " AND event = ?"
            params.append(event)
        if bet_type:
            query += " AND bet_type = ?"
            params.append(bet_type)
        if sportsbook:
            query += " AND sportsbook = ?"
            params.append(sportsbook)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return self._conn.execute(query, params).fetchall()

    def get_latest_for_event(self, event: str, bet_type: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            """SELECT * FROM lines
               WHERE event = ? AND bet_type = ?
               AND id IN (
                   SELECT MAX(id) FROM lines
                   WHERE event = ? AND bet_type = ?
                   GROUP BY sportsbook
               )
               ORDER BY sportsbook""",
            (event, bet_type, event, bet_type),
        ).fetchall()

    def get_events(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT DISTINCT event FROM lines ORDER BY event"
        ).fetchall()

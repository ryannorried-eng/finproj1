"""Repository for ``events`` table."""

from __future__ import annotations

import sqlite3


class EventsRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_many_upsert(self, events: list[tuple]) -> int:
        """Upsert event rows.

        Tuple shape:
            (api_event_id, sport, commence_time, home_team, away_team,
             event_display)

        On conflict updates ``last_seen_at`` and team/display fields.
        Returns the number of rows affected.
        """
        if not events:
            return 0
        cursor = self._conn.executemany(
            """INSERT INTO events
               (api_event_id, sport, commence_time, home_team, away_team,
                event_display)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(api_event_id) DO UPDATE SET
                   last_seen_at  = CURRENT_TIMESTAMP,
                   home_team     = excluded.home_team,
                   away_team     = excluded.away_team,
                   event_display = excluded.event_display,
                   commence_time = excluded.commence_time,
                   sport         = excluded.sport""",
            events,
        )
        return cursor.rowcount

    def list_events(
        self,
        sport: str | None = None,
        since_iso: str | None = None,
    ) -> list[dict]:
        """Return events, optionally filtered by sport and/or last_seen_at.

        Results are sorted by commence_time descending.
        """
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []
        if sport is not None:
            query += " AND sport = ?"
            params.append(sport)
        if since_iso is not None:
            query += " AND last_seen_at >= ?"
            params.append(since_iso)
        query += " ORDER BY commence_time DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "api_event_id": r["api_event_id"],
                "event_display": r["event_display"],
                "commence_time": r["commence_time"],
                "sport": r["sport"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
            }
            for r in rows
        ]

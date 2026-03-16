"""Repository for `lines` table queries/writes."""

from __future__ import annotations

import sqlite3


class LinesRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def insert_many(self, rows: list[tuple]) -> int:
        """Insert or upsert rows into lines table.

        Expects 13-element tuples:
            (sportsbook, sport, event, bet_type, home_team, away_team,
             home_value, away_value, home_price, away_price, timestamp,
             commence_time, api_event_id)

        When api_event_id is non-NULL, uses UPSERT on the
        ``uq_lines_snapshot`` unique index to update mutable price/value
        columns.  Rows without api_event_id fall back to plain INSERT.
        """
        upsert_rows = [r for r in rows if r[12] is not None]
        plain_rows = [r for r in rows if r[12] is None]
        count = 0

        if upsert_rows:
            cursor = self._conn.executemany(
                """INSERT INTO lines
                   (sportsbook, sport, event, bet_type, home_team, away_team,
                    home_value, away_value, home_price, away_price, timestamp,
                    commence_time, api_event_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(api_event_id, bet_type, sportsbook, timestamp)
                   DO UPDATE SET
                       home_price  = excluded.home_price,
                       away_price  = excluded.away_price,
                       home_value  = excluded.home_value,
                       away_value  = excluded.away_value,
                       event       = excluded.event,
                       home_team   = excluded.home_team,
                       away_team   = excluded.away_team,
                       sport       = excluded.sport,
                       commence_time = excluded.commence_time""",
                upsert_rows,
            )
            count += cursor.rowcount

        if plain_rows:
            cursor = self._conn.executemany(
                """INSERT INTO lines
                   (sportsbook, sport, event, bet_type, home_team, away_team,
                    home_value, away_value, home_price, away_price, timestamp,
                    commence_time, api_event_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                plain_rows,
            )
            count += cursor.rowcount

        return count

    # ── Query methods ─────────────────────────────────────────────────

    def get_latest_for_event(self, event: str, bet_type: str) -> list[sqlite3.Row]:
        """Latest line per sportsbook for an event+bet_type.

        Uses a window function instead of a correlated MAX(id) subquery.
        """
        return self._conn.execute(
            """SELECT * FROM (
                   SELECT *, ROW_NUMBER() OVER (
                       PARTITION BY sportsbook
                       ORDER BY timestamp DESC, id DESC
                   ) AS rn
                   FROM lines
                   WHERE event = ? AND bet_type = ?
               ) WHERE rn = 1
               ORDER BY sportsbook""",
            (event, bet_type),
        ).fetchall()

    def get_latest_for_api_event(
        self, api_event_id: str, bet_type: str
    ) -> list[sqlite3.Row]:
        """Latest line per sportsbook using stable api_event_id."""
        return self._conn.execute(
            """SELECT * FROM (
                   SELECT *, ROW_NUMBER() OVER (
                       PARTITION BY sportsbook
                       ORDER BY timestamp DESC, id DESC
                   ) AS rn
                   FROM lines
                   WHERE api_event_id = ? AND bet_type = ?
               ) WHERE rn = 1
               ORDER BY sportsbook""",
            (api_event_id, bet_type),
        ).fetchall()

    def get_latest_for_sport(self, sport: str) -> list[sqlite3.Row]:
        """Latest line per (api_event_id, bet_type, sportsbook) for a sport.

        Returns the most recent line for every sportsbook that has ever
        reported on each event, regardless of which API fetch it came from.
        """
        return self._conn.execute(
            """SELECT * FROM (
                   SELECT *, ROW_NUMBER() OVER (
                       PARTITION BY api_event_id, bet_type, sportsbook
                       ORDER BY timestamp DESC, id DESC
                   ) AS rn
                   FROM lines
                   WHERE sport = ? AND api_event_id IS NOT NULL
               ) WHERE rn = 1
               ORDER BY api_event_id, bet_type, sportsbook""",
            (sport,),
        ).fetchall()

    def get_recent_lines(self, limit: int = 10000) -> list[sqlite3.Row]:
        """Fetch recent lines with only needed columns, ordered by timestamp DESC."""
        return self._conn.execute(
            """SELECT sportsbook, sport, event, bet_type, home_team, away_team,
                      home_value, away_value, home_price, away_price,
                      timestamp, commence_time, api_event_id, ingested_at
               FROM lines
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    def get_lines(
        self,
        event: str | None = None,
        bet_type: str | None = None,
        sportsbook: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        """Dynamic query builder — filters by event/bet_type/sportsbook."""
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

    def get_events(self) -> list[sqlite3.Row]:
        """Distinct event names, using idx_lines_event index."""
        return self._conn.execute(
            "SELECT DISTINCT event FROM lines ORDER BY event"
        ).fetchall()

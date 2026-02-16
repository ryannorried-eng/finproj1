"""SQLite persistence for storing and querying historical betting lines."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from line_tracker.models import BettingLine, BetType

DEFAULT_DB_PATH = Path("lines.db")


class LineStore:
    """Stores betting lines in SQLite for historical tracking."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sportsbook TEXT NOT NULL,
                sport TEXT NOT NULL,
                event TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_value REAL NOT NULL,
                away_value REAL NOT NULL,
                home_price REAL,
                away_price REAL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                commence_time TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type
            ON lines (event, bet_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON lines (timestamp)
        """)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that may be missing on older databases."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(lines)").fetchall()
        }
        if "commence_time" not in cols:
            self._conn.execute(
                "ALTER TABLE lines ADD COLUMN commence_time TEXT"
            )
            self._conn.commit()

    def save_lines(self, lines: list[BettingLine]) -> int:
        """Save a batch of lines. Returns number of rows inserted."""
        rows = [
            (
                ln.sportsbook,
                ln.sport,
                ln.event,
                ln.bet_type.value,
                ln.home_team,
                ln.away_team,
                ln.home_value,
                ln.away_value,
                ln.home_price,
                ln.away_price,
                ln.timestamp.isoformat(),
                ln.commence_time.isoformat() if ln.commence_time else None,
            )
            for ln in lines
        ]
        self._conn.executemany(
            """INSERT INTO lines
               (sportsbook, sport, event, bet_type, home_team, away_team,
                home_value, away_value, home_price, away_price, timestamp,
                commence_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_lines(
        self,
        event: str | None = None,
        bet_type: BetType | None = None,
        sportsbook: str | None = None,
        limit: int = 100,
    ) -> list[BettingLine]:
        """Query stored lines with optional filters."""
        query = "SELECT * FROM lines WHERE 1=1"
        params: list = []
        if event:
            query += " AND event = ?"
            params.append(event)
        if bet_type:
            query += " AND bet_type = ?"
            params.append(bet_type.value)
        if sportsbook:
            query += " AND sportsbook = ?"
            params.append(sportsbook)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_line(row) for row in rows]

    def get_latest_for_event(
        self, event: str, bet_type: BetType
    ) -> list[BettingLine]:
        """Get the most recent line per sportsbook for an event+bet_type."""
        rows = self._conn.execute(
            """SELECT * FROM lines
               WHERE event = ? AND bet_type = ?
               AND id IN (
                   SELECT MAX(id) FROM lines
                   WHERE event = ? AND bet_type = ?
                   GROUP BY sportsbook
               )
               ORDER BY sportsbook""",
            (event, bet_type.value, event, bet_type.value),
        ).fetchall()
        return [_row_to_line(row) for row in rows]

    def get_close_lines(
        self,
        event: str,
        bet_type: BetType,
        before: datetime,
        sportsbook: str | None = None,
    ) -> list[BettingLine]:
        """Get the last snapshot per sportsbook before *before* time.

        For closing-line lookups: pass ``commence_time`` as *before* to
        obtain each book's last line prior to game start.  Optionally
        filter to a single *sportsbook* for same-book close.
        """
        before_iso = before.isoformat()
        if sportsbook:
            rows = self._conn.execute(
                """SELECT * FROM lines
                   WHERE event = ? AND bet_type = ? AND sportsbook = ?
                         AND timestamp <= ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (event, bet_type.value, sportsbook, before_iso),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM lines
                   WHERE event = ? AND bet_type = ? AND timestamp <= ?
                         AND id IN (
                             SELECT MAX(id) FROM lines
                             WHERE event = ? AND bet_type = ?
                                   AND timestamp <= ?
                             GROUP BY sportsbook
                         )
                   ORDER BY sportsbook""",
                (
                    event, bet_type.value, before_iso,
                    event, bet_type.value, before_iso,
                ),
            ).fetchall()
        return [_row_to_line(row) for row in rows]

    def get_events(self) -> list[str]:
        """List all distinct events in the database."""
        rows = self._conn.execute(
            "SELECT DISTINCT event FROM lines ORDER BY event"
        ).fetchall()
        return [row["event"] for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _row_to_line(row: sqlite3.Row) -> BettingLine:
    ct_raw = row["commence_time"]
    commence_time = (
        datetime.fromisoformat(ct_raw).replace(tzinfo=timezone.utc)
        if ct_raw
        else None
    )
    return BettingLine(
        sportsbook=row["sportsbook"],
        sport=row["sport"],
        event=row["event"],
        bet_type=BetType(row["bet_type"]),
        home_team=row["home_team"],
        away_team=row["away_team"],
        home_value=row["home_value"],
        away_value=row["away_value"],
        home_price=row["home_price"],
        away_price=row["away_price"],
        timestamp=datetime.fromisoformat(row["timestamp"]).replace(
            tzinfo=timezone.utc
        ),
        commence_time=commence_time,
    )

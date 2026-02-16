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
            CREATE TABLE IF NOT EXISTS bet_clv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_id TEXT NOT NULL,
                leg_index INTEGER NOT NULL,
                sport TEXT,
                market TEXT,
                pick_sportsbook TEXT,
                event_name TEXT,
                selection TEXT,
                pick_line REAL,
                pick_odds REAL,
                pick_timestamp TEXT,
                commence_time TEXT,
                edge_pct REAL,
                edge_z REAL,
                books_used INTEGER,
                market_hold_median REAL,
                market_volatility_sigma REAL,
                confidence TEXT,
                quality_tier TEXT,
                close_timestamp TEXT,
                close_estimated INTEGER DEFAULT 0,
                exec_close_odds REAL,
                market_close_odds REAL,
                exec_close_decimal REAL,
                market_close_decimal REAL,
                exec_clv_prob REAL,
                market_clv_prob REAL,
                beat_close_exec INTEGER,
                beat_close_market INTEGER,
                settled_at TEXT,
                outcome TEXT,
                pick_lines_max_ts TEXT,
                pick_lines_count INTEGER,
                UNIQUE(bet_id, leg_index)
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
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_clv_bet
            ON bet_clv (bet_id)
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

        # Provenance columns on bet_clv
        clv_cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(bet_clv)"
            ).fetchall()
        }
        for col in ("pick_lines_max_ts", "pick_lines_count"):
            if col not in clv_cols:
                col_type = "TEXT" if col.endswith("_ts") else "INTEGER"
                self._conn.execute(
                    f"ALTER TABLE bet_clv ADD COLUMN {col} {col_type}"
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

    def get_lines_asof(
        self,
        event: str,
        bet_type: BetType,
        asof: datetime,
    ) -> list[BettingLine]:
        """Get the latest line per sportsbook as of *asof* timestamp.

        Only lines with ``timestamp <= asof`` are considered.  Returns
        one line per sportsbook (the most recent before the cutoff).
        """
        asof_iso = asof.isoformat()
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
                event, bet_type.value, asof_iso,
                event, bet_type.value, asof_iso,
            ),
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

    # ------------------------------------------------------------------
    # CLV analytics persistence
    # ------------------------------------------------------------------

    def save_clv_rows(self, rows: list[dict]) -> int:
        """Upsert CLV analytics rows (one per settled leg)."""
        sql = """
            INSERT OR REPLACE INTO bet_clv (
                bet_id, leg_index, sport, market, pick_sportsbook,
                event_name, selection, pick_line, pick_odds,
                pick_timestamp, commence_time,
                edge_pct, edge_z, books_used,
                market_hold_median, market_volatility_sigma,
                confidence, quality_tier,
                close_timestamp, close_estimated,
                exec_close_odds, market_close_odds,
                exec_close_decimal, market_close_decimal,
                exec_clv_prob, market_clv_prob,
                beat_close_exec, beat_close_market,
                settled_at, outcome,
                pick_lines_max_ts, pick_lines_count
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?
            )
        """
        params = []
        for r in rows:
            params.append((
                r["bet_id"], r["leg_index"], r.get("sport"),
                r.get("market"), r.get("pick_sportsbook"),
                r.get("event_name"), r.get("selection"),
                r.get("pick_line"), r.get("pick_odds"),
                r.get("pick_timestamp"), r.get("commence_time"),
                r.get("edge_pct"), r.get("edge_z"),
                r.get("books_used"),
                r.get("market_hold_median"),
                r.get("market_volatility_sigma"),
                r.get("confidence"), r.get("quality_tier"),
                r.get("close_timestamp"),
                1 if r.get("close_estimated") else 0,
                r.get("exec_close_odds"),
                r.get("market_close_odds"),
                r.get("exec_close_decimal"),
                r.get("market_close_decimal"),
                r.get("exec_clv_prob"), r.get("market_clv_prob"),
                r.get("beat_close_exec"),
                r.get("beat_close_market"),
                r.get("settled_at"), r.get("outcome"),
                r.get("pick_lines_max_ts"),
                r.get("pick_lines_count"),
            ))
        self._conn.executemany(sql, params)
        self._conn.commit()
        return len(params)

    def get_clv_rows(
        self,
        start: str | None = None,
        end: str | None = None,
        sport: str | None = None,
        market: str | None = None,
        book: str | None = None,
        confidence: str | None = None,
        tier: str | None = None,
        include_estimated: bool = False,
    ) -> list[dict]:
        """Fetch CLV analytics rows with optional filters."""
        query = "SELECT * FROM bet_clv WHERE 1=1"
        params: list = []
        if not include_estimated:
            query += " AND close_estimated = 0"
        if start:
            query += " AND settled_at >= ?"
            params.append(start)
        if end:
            query += " AND settled_at <= ?"
            params.append(end)
        if sport:
            query += " AND sport = ?"
            params.append(sport)
        if market:
            query += " AND market = ?"
            params.append(market)
        if book:
            query += " AND pick_sportsbook = ?"
            params.append(book)
        if confidence:
            query += " AND confidence = ?"
            params.append(confidence)
        if tier:
            query += " AND quality_tier = ?"
            params.append(tier)
        query += " ORDER BY settled_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

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

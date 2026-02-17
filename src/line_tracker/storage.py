"""SQLite persistence for storing and querying historical betting lines."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

from line_tracker.models import BettingLine, BetType

DEFAULT_DB_PATH = Path.home() / ".line_tracker" / "lines.db"
LEGACY_DB_PATH = Path("lines.db")


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve DB path with one-time legacy migration.

    If using the default path and the new DB doesn't exist yet but a legacy
    local ./lines.db exists, copy it into ~/.line_tracker/lines.db.
    This is safe and idempotent: existing new DB is never overwritten.
    """
    target = DEFAULT_DB_PATH if db_path is None else Path(db_path)
    if target != DEFAULT_DB_PATH:
        return target

    if target.exists() or not LEGACY_DB_PATH.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(LEGACY_DB_PATH, target)
    return target


class LineStore:
    """Stores betting lines in SQLite for historical tracking."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = resolve_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bet_clv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_id TEXT NOT NULL,
                leg_index INTEGER NOT NULL DEFAULT 0,
                event TEXT NOT NULL,
                market TEXT NOT NULL,
                pick_side TEXT NOT NULL,
                pick_line_value REAL,
                pick_odds_american REAL NOT NULL,
                pick_odds_decimal REAL NOT NULL,
                consensus_prob_at_pick REAL NOT NULL,
                market_hold_median_at_pick REAL DEFAULT 0.0,
                market_volatility_sigma_at_pick REAL DEFAULT 0.0,
                consensus_prob_close REAL,
                best_odds_close_american REAL,
                best_odds_close_decimal REAL,
                closed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(bet_id, leg_index)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bet_clv_bet_id
            ON bet_clv (bet_id)
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
            CREATE INDEX IF NOT EXISTS idx_sportsbook
            ON lines (sportsbook)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type_book
            ON lines (event, bet_type, sportsbook)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_thresholds (
                key TEXT PRIMARY KEY,
                json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                bet_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                stake REAL NOT NULL,
                total_odds_american INTEGER NOT NULL,
                total_odds_decimal REAL NOT NULL,
                potential_payout REAL NOT NULL,
                profit REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                settled_at TEXT,
                outcome TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bet_legs (
                leg_id TEXT PRIMARY KEY,
                bet_id TEXT NOT NULL REFERENCES bets(bet_id),
                sport TEXT,
                market TEXT,
                event_name TEXT,
                selection TEXT,
                line_value REAL,
                odds_american INTEGER NOT NULL,
                odds_decimal REAL NOT NULL,
                sportsbook TEXT,
                pick_timestamp TEXT,
                commence_time TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bet_legs_bet_id
            ON bet_legs (bet_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bets_status
            ON bets (status)
        """)
        self._conn.commit()
        self._migrate_commence_time()
        self._migrate_clv_metadata()

    def _migrate_commence_time(self) -> None:
        """Add commence_time column if it doesn't exist yet."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(lines)").fetchall()
        }
        if "commence_time" not in cols:
            self._conn.execute(
                "ALTER TABLE lines ADD COLUMN commence_time TEXT"
            )
            self._conn.commit()

    def _migrate_clv_metadata(self) -> None:
        """Add pick-time metadata columns to bet_clv if they don't exist."""
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(bet_clv)"
            ).fetchall()
        }
        new_cols = {
            "pick_sportsbook": "TEXT",
            "sport": "TEXT",
            "confidence_at_pick": "TEXT",
            "quality_tier_at_pick": "TEXT",
            "edge_pct_at_pick": "REAL",
            "edge_z_at_pick": "REAL",
            "books_used_at_pick": "INTEGER",
            "agreement_score_at_pick": "REAL",
        }
        for col, col_type in new_cols.items():
            if col not in cols:
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
        cursor = self._conn.executemany(
            """INSERT INTO lines
               (sportsbook, sport, event, bet_type, home_team, away_team,
                home_value, away_value, home_price, away_price, timestamp,
                commence_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        return cursor.rowcount

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

    def get_events(self) -> list[str]:
        """List all distinct events in the database."""
        rows = self._conn.execute(
            "SELECT DISTINCT event FROM lines ORDER BY event"
        ).fetchall()
        return [row["event"] for row in rows]

    # ------------------------------------------------------------------
    # CLV tracking
    # ------------------------------------------------------------------

    def save_clv_pick(
        self,
        *,
        bet_id: str,
        leg_index: int,
        event: str,
        market: str,
        pick_side: str,
        pick_line_value: float | None,
        pick_odds_american: float,
        pick_odds_decimal: float,
        consensus_prob_at_pick: float,
        market_hold_median_at_pick: float = 0.0,
        market_volatility_sigma_at_pick: float = 0.0,
        pick_sportsbook: str | None = None,
        sport: str | None = None,
        confidence_at_pick: str | None = None,
        quality_tier_at_pick: str | None = None,
        edge_pct_at_pick: float | None = None,
        edge_z_at_pick: float | None = None,
        books_used_at_pick: int | None = None,
        agreement_score_at_pick: float | None = None,
    ) -> None:
        """Persist the pick-time snapshot for one leg of a bet."""
        self._conn.execute(
            """INSERT OR REPLACE INTO bet_clv
               (bet_id, leg_index, event, market, pick_side,
                pick_line_value, pick_odds_american,
                pick_odds_decimal, consensus_prob_at_pick,
                market_hold_median_at_pick,
                market_volatility_sigma_at_pick,
                pick_sportsbook, sport, confidence_at_pick,
                quality_tier_at_pick, edge_pct_at_pick,
                edge_z_at_pick, books_used_at_pick,
                agreement_score_at_pick)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bet_id,
                leg_index,
                event,
                market,
                pick_side,
                pick_line_value,
                pick_odds_american,
                pick_odds_decimal,
                consensus_prob_at_pick,
                market_hold_median_at_pick,
                market_volatility_sigma_at_pick,
                pick_sportsbook,
                sport,
                confidence_at_pick,
                quality_tier_at_pick,
                edge_pct_at_pick,
                edge_z_at_pick,
                books_used_at_pick,
                agreement_score_at_pick,
            ),
        )
        self._conn.commit()

    def close_clv(
        self,
        *,
        bet_id: str,
        leg_index: int,
        consensus_prob_close: float,
        best_odds_close_american: float,
        best_odds_close_decimal: float,
    ) -> None:
        """Write the closing-line snapshot for one leg."""
        self._conn.execute(
            """UPDATE bet_clv
               SET consensus_prob_close = ?,
                   best_odds_close_american = ?,
                   best_odds_close_decimal = ?,
                   closed_at = CURRENT_TIMESTAMP
               WHERE bet_id = ? AND leg_index = ?""",
            (
                consensus_prob_close,
                best_odds_close_american,
                best_odds_close_decimal,
                bet_id,
                leg_index,
            ),
        )
        self._conn.commit()

    def get_clv(self, bet_id: str) -> list[dict]:
        """Return all CLV rows for a bet, ordered by leg_index."""
        rows = self._conn.execute(
            "SELECT * FROM bet_clv WHERE bet_id = ? ORDER BY leg_index",
            (bet_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_clv(self) -> list[dict]:
        """Return every CLV row that has been closed (has closing data)."""
        rows = self._conn.execute(
            "SELECT * FROM bet_clv WHERE closed_at IS NOT NULL "
            "ORDER BY closed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Bet persistence
    # ------------------------------------------------------------------

    def insert_bet(self, bet_row: dict) -> str:
        """Persist a placed bet. Returns the bet_id."""
        self._conn.execute(
            """INSERT INTO bets
               (bet_id, created_at, sportsbook, stake,
                total_odds_american, total_odds_decimal,
                potential_payout, profit, status, settled_at, outcome)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bet_row["bet_id"],
                bet_row["created_at"],
                bet_row["sportsbook"],
                bet_row["stake"],
                bet_row["total_odds_american"],
                bet_row["total_odds_decimal"],
                bet_row["potential_payout"],
                bet_row["profit"],
                bet_row.get("status", "active"),
                bet_row.get("settled_at"),
                bet_row.get("outcome"),
            ),
        )
        self._conn.commit()
        return bet_row["bet_id"]

    def insert_legs(self, bet_id: str, legs: list[dict]) -> None:
        """Persist all legs for a bet in one batch."""
        rows = [
            (
                lg["leg_id"],
                bet_id,
                lg.get("sport"),
                lg.get("market"),
                lg.get("event_name"),
                lg.get("selection"),
                lg.get("line_value"),
                lg["odds_american"],
                lg["odds_decimal"],
                lg.get("sportsbook"),
                lg.get("pick_timestamp"),
                lg.get("commence_time"),
            )
            for lg in legs
        ]
        self._conn.executemany(
            """INSERT INTO bet_legs
               (leg_id, bet_id, sport, market, event_name,
                selection, line_value, odds_american, odds_decimal,
                sportsbook, pick_timestamp, commence_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def insert_bet_with_legs(
        self, bet_row: dict, legs: list[dict],
    ) -> str:
        """Persist a bet + legs in a single transaction. Returns bet_id."""
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """INSERT INTO bets
                   (bet_id, created_at, sportsbook, stake,
                    total_odds_american, total_odds_decimal,
                    potential_payout, profit, status, settled_at, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    bet_row["bet_id"],
                    bet_row["created_at"],
                    bet_row["sportsbook"],
                    bet_row["stake"],
                    bet_row["total_odds_american"],
                    bet_row["total_odds_decimal"],
                    bet_row["potential_payout"],
                    bet_row["profit"],
                    bet_row.get("status", "active"),
                    bet_row.get("settled_at"),
                    bet_row.get("outcome"),
                ),
            )
            for lg in legs:
                self._conn.execute(
                    """INSERT INTO bet_legs
                       (leg_id, bet_id, sport, market, event_name,
                        selection, line_value, odds_american, odds_decimal,
                        sportsbook, pick_timestamp, commence_time)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        lg["leg_id"],
                        bet_row["bet_id"],
                        lg.get("sport"),
                        lg.get("market"),
                        lg.get("event_name"),
                        lg.get("selection"),
                        lg.get("line_value"),
                        lg["odds_american"],
                        lg["odds_decimal"],
                        lg.get("sportsbook"),
                        lg.get("pick_timestamp"),
                        lg.get("commence_time"),
                    ),
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return bet_row["bet_id"]

    def get_bets(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Fetch bets, optionally filtered by status."""
        query = "SELECT * FROM bets WHERE 1=1"
        params: list = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_bet_legs(self, bet_id: str) -> list[dict]:
        """Fetch all legs for a given bet."""
        rows = self._conn.execute(
            "SELECT * FROM bet_legs WHERE bet_id = ? ORDER BY leg_id",
            (bet_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def settle_bet_db(
        self,
        bet_id: str,
        outcome: str,
        settled_at: str,
        profit: float,
        potential_payout: float,
    ) -> None:
        """Update a bet's status to settled with outcome and final P&L."""
        self._conn.execute(
            """UPDATE bets
               SET status = ?, outcome = ?, settled_at = ?,
                   profit = ?, potential_payout = ?
               WHERE bet_id = ?""",
            (outcome, outcome, settled_at, profit, potential_payout, bet_id),
        )
        self._conn.commit()

    def delete_bet_db(self, bet_id: str) -> None:
        """Delete a bet and its legs from the database."""
        self._conn.execute(
            "DELETE FROM bet_legs WHERE bet_id = ?", (bet_id,),
        )
        self._conn.execute(
            "DELETE FROM bets WHERE bet_id = ?", (bet_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Calibration thresholds
    # ------------------------------------------------------------------

    def save_calibration(self, key: str, json_str: str) -> None:
        """Persist a calibration result keyed by scope (e.g. 'global')."""
        self._conn.execute(
            """INSERT OR REPLACE INTO calibration_thresholds
               (key, json, created_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (key, json_str),
        )
        self._conn.commit()

    def load_calibration(self, key: str) -> str | None:
        """Load a calibration JSON string by key, or None."""
        row = self._conn.execute(
            "SELECT json FROM calibration_thresholds WHERE key = ?",
            (key,),
        ).fetchone()
        return row["json"] if row else None

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

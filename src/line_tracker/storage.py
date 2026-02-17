"""SQLite persistence for storing and querying historical betting lines."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

from line_tracker.db.migrate import ensure_latest
from line_tracker.db.repos import BetsRepo, CalibrationRepo, ClvRepo, LinesRepo
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
        self._in_explicit_txn = False
        self._txn_depth = 0
        self._configure_connection()
        migrations_path = Path(__file__).parent / "db" / "migrations"
        ensure_latest(self._conn, migrations_path)

        self.lines_repo = LinesRepo(self._conn)
        self.bets_repo = BetsRepo(self._conn)
        self.clv_repo = ClvRepo(self._conn)
        self.calibration_repo = CalibrationRepo(self._conn)

    def _maybe_commit(self) -> None:
        if not self._in_explicit_txn:
            self._conn.commit()

    def _configure_connection(self) -> None:
        """Apply defensive SQLite connection settings."""
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")

    @contextmanager
    def transaction(self):
        """Run statements in an explicit transaction with rollback safety."""
        nested = self._txn_depth > 0
        savepoint = f"sp_{self._txn_depth + 1}"
        if nested:
            self._conn.execute(f"SAVEPOINT {savepoint}")
        else:
            self._conn.execute("BEGIN IMMEDIATE")
            self._in_explicit_txn = True
        self._txn_depth += 1
        try:
            yield
        except Exception:
            if nested:
                self._conn.execute(f"ROLLBACK TO {savepoint}")
                self._conn.execute(f"RELEASE {savepoint}")
            else:
                self._conn.execute("ROLLBACK")
            raise
        else:
            if nested:
                self._conn.execute(f"RELEASE {savepoint}")
            else:
                self._conn.execute("COMMIT")
        finally:
            self._txn_depth -= 1
            if self._txn_depth == 0:
                self._in_explicit_txn = False

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
        count = self.lines_repo.insert_many(rows)
        self._maybe_commit()
        return count

    def get_lines(
        self,
        event: str | None = None,
        bet_type: BetType | None = None,
        sportsbook: str | None = None,
        limit: int = 100,
    ) -> list[BettingLine]:
        """Query stored lines with optional filters."""
        rows = self.lines_repo.get_lines(
            event=event,
            bet_type=bet_type.value if bet_type else None,
            sportsbook=sportsbook,
            limit=limit,
        )
        return [_row_to_line(row) for row in rows]

    def get_latest_for_event(
        self,
        event: str,
        bet_type: BetType,
    ) -> list[BettingLine]:
        """Get the most recent line per sportsbook for an event+bet_type."""
        rows = self.lines_repo.get_latest_for_event(event, bet_type.value)
        return [_row_to_line(row) for row in rows]

    def get_events(self) -> list[str]:
        """List all distinct events in the database."""
        rows = self.lines_repo.get_events()
        return [row["event"] for row in rows]

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
        self.clv_repo.save_pick(
            bet_id=bet_id,
            leg_index=leg_index,
            event=event,
            market=market,
            pick_side=pick_side,
            pick_line_value=pick_line_value,
            pick_odds_american=pick_odds_american,
            pick_odds_decimal=pick_odds_decimal,
            consensus_prob_at_pick=consensus_prob_at_pick,
            market_hold_median_at_pick=market_hold_median_at_pick,
            market_volatility_sigma_at_pick=market_volatility_sigma_at_pick,
            pick_sportsbook=pick_sportsbook,
            sport=sport,
            confidence_at_pick=confidence_at_pick,
            quality_tier_at_pick=quality_tier_at_pick,
            edge_pct_at_pick=edge_pct_at_pick,
            edge_z_at_pick=edge_z_at_pick,
            books_used_at_pick=books_used_at_pick,
            agreement_score_at_pick=agreement_score_at_pick,
        )
        self._maybe_commit()

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
        self.clv_repo.close_leg(
            bet_id=bet_id,
            leg_index=leg_index,
            consensus_prob_close=consensus_prob_close,
            best_odds_close_american=best_odds_close_american,
            best_odds_close_decimal=best_odds_close_decimal,
        )
        self._maybe_commit()

    def get_clv(self, bet_id: str) -> list[dict]:
        """Return all CLV rows for a bet, ordered by leg_index."""
        return self.clv_repo.get_for_bet(bet_id)

    def get_all_clv(self) -> list[dict]:
        """Return every CLV row that has been closed (has closing data)."""
        return self.clv_repo.get_all_closed()

    def insert_bet(self, bet_row: dict) -> str:
        """Persist a placed bet. Returns the bet_id."""
        self.bets_repo.insert_bet(bet_row)
        self._maybe_commit()
        return bet_row["bet_id"]

    def insert_legs(self, bet_id: str, legs: list[dict]) -> None:
        """Persist all legs for a bet in one batch."""
        self.bets_repo.insert_legs(bet_id, legs)
        self._maybe_commit()

    def insert_bet_with_legs(self, bet_row: dict, legs: list[dict]) -> str:
        """Persist a bet + legs in a single transaction. Returns bet_id."""
        with self.transaction():
            self.bets_repo.insert_bet(bet_row)
            self.bets_repo.insert_legs(bet_row["bet_id"], legs)
        return bet_row["bet_id"]

    def get_bets(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Fetch bets, optionally filtered by status."""
        return self.bets_repo.get_bets(status=status, limit=limit)

    def get_bet_legs(self, bet_id: str) -> list[dict]:
        """Fetch all legs for a given bet."""
        return self.bets_repo.get_bet_legs(bet_id)

    def settle_bet_db(
        self,
        bet_id: str,
        outcome: str,
        settled_at: str,
        profit: float,
        potential_payout: float,
    ) -> None:
        """Update a bet's status to settled with outcome and final P&L."""
        self.bets_repo.settle_bet(
            bet_id=bet_id,
            outcome=outcome,
            settled_at=settled_at,
            profit=profit,
            potential_payout=potential_payout,
        )
        self._maybe_commit()

    def delete_bet_db(self, bet_id: str) -> None:
        """Delete a bet and its legs from the database."""
        self.bets_repo.delete_bet(bet_id)
        self._maybe_commit()

    def save_calibration(self, key: str, json_str: str) -> None:
        """Persist a calibration result keyed by scope (e.g. 'global')."""
        self.calibration_repo.save(key, json_str)
        self._maybe_commit()

    def load_calibration(self, key: str) -> str | None:
        """Load a calibration JSON string by key, or None."""
        return self.calibration_repo.load(key)

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
            tzinfo=timezone.utc,
        ),
        commence_time=commence_time,
    )

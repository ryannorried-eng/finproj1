"""Tests for SQLite schema versioning and additive migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from line_tracker.db.migrate import ensure_latest, get_schema_version
from line_tracker.storage import LineStore


def _migrations_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "src" / "line_tracker" / "db" / "migrations"
    )


def _latest_version() -> int:
    """Derive the latest migration version from filenames."""
    mpath = _migrations_path()
    versions = []
    for p in mpath.glob("*.sql"):
        prefix = p.stem.split("_", 1)[0]
        if prefix.isdigit():
            versions.append(int(prefix))
    return max(versions) if versions else 0


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _index_exists(conn: sqlite3.Connection, index_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def test_linestore_init_runs_migrations_to_latest(tmp_path):
    db = tmp_path / "migrations_a.db"
    latest = _latest_version()
    with LineStore(db_path=db) as store:
        version = get_schema_version(store._conn)
        assert version == latest

        for table in (
            "schema_version",
            "lines",
            "bet_clv",
            "calibration_thresholds",
            "bets",
            "bet_legs",
            "events",
            "published_slates",
            "slate_picks",
            "rec_snapshots",
            "outcomes",
            "clv_model_scores",
        ):
            assert _table_exists(store._conn, table)


def test_ensure_latest_is_idempotent_noop_on_latest(tmp_path):
    db = tmp_path / "migrations_b.db"
    migrations = _migrations_path()
    latest = _latest_version()

    with sqlite3.connect(db) as conn:
        ensure_latest(conn, migrations)
        assert get_schema_version(conn) == latest

        ensure_latest(conn, migrations)
        assert get_schema_version(conn) == latest


def test_schema_version_one_applies_remaining_migrations(tmp_path):
    db = tmp_path / "migrations_c.db"
    migrations = _migrations_path()
    latest = _latest_version()

    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version(version) VALUES (1)")

        init_sql = (migrations / "0001_init.sql").read_text(encoding="utf-8")
        conn.executescript(init_sql)
        conn.commit()

        ensure_latest(conn, migrations)
        assert get_schema_version(conn) == latest

        for index_name in (
            "idx_bet_clv_bet_id",
            "idx_event_type",
            "idx_timestamp",
            "idx_sportsbook",
            "idx_lines_event_type_book_id",
            "idx_bet_legs_bet_id",
            "idx_bets_status",
            "idx_events_sport_time",
        ):
            assert _index_exists(conn, index_name)


def test_fresh_db_has_rec_snapshots_with_all_columns(tmp_path):
    """A brand-new DB runs ALL migrations strictly and ends with rec_snapshots
    containing columns added by later migrations (outcome_result, actual_roi,
    run_id)."""
    db = tmp_path / "fresh.db"
    migrations = _migrations_path()
    latest = _latest_version()

    with sqlite3.connect(db) as conn:
        ensure_latest(conn, migrations)
        assert get_schema_version(conn) == latest
        assert _table_exists(conn, "rec_snapshots")

        expected_cols = {
            "snapshot_id", "created_at", "event_id", "sport", "market",
            "selection", "book", "line", "odds_american", "odds_decimal",
            "consensus_prob", "tier", "close_odds_american",
            "close_odds_decimal", "close_implied_prob", "open_implied_prob",
            "clv_delta_american", "clv_delta_implied", "closed_at",
            "outcome_result", "actual_roi", "run_id",
        }
        actual_cols = _column_names(conn, "rec_snapshots")
        assert expected_cols.issubset(actual_cols), (
            f"Missing columns: {expected_cols - actual_cols}"
        )

        for idx in ("idx_rec_snap_created", "idx_rec_snap_run_id"):
            assert _index_exists(conn, idx)

"""UI diagnostics helpers for DB status/count/timestamps."""

from __future__ import annotations

from line_tracker.db.migrate import get_schema_version


def get_db_status(store) -> dict:
    conn = store._conn
    return {
        "db_path": str(store.db_path),
        "schema_version": get_schema_version(conn),
        "pragmas": {
            "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
            "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
            "busy_timeout": conn.execute("PRAGMA busy_timeout").fetchone()[0],
        },
    }


def get_db_counts(store) -> dict:
    conn = store._conn
    tables = ["lines", "bets", "bet_legs", "bet_clv", "calibration_thresholds"]
    out: dict[str, int] = {}
    for table in tables:
        out[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return out


def get_latest_timestamps(store) -> dict:
    conn = store._conn
    return {
        "latest_line_timestamp": conn.execute("SELECT MAX(timestamp) FROM lines").fetchone()[0],
        "latest_bet_created_at": conn.execute("SELECT MAX(created_at) FROM bets").fetchone()[0],
        "latest_clv_closed_at": conn.execute("SELECT MAX(closed_at) FROM bet_clv").fetchone()[0],
    }

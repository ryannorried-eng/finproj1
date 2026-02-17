"""SQLite schema versioning and additive migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (0)")


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return current integer schema version, initializing metadata if needed."""
    _ensure_schema_version_table(conn)
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    return int(row[0])


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set schema version in metadata table."""
    _ensure_schema_version_table(conn)
    conn.execute("UPDATE schema_version SET version = ?", (int(version),))


def _discover_migrations(migrations_path: Path) -> list[tuple[int, Path]]:
    migrations: list[tuple[int, Path]] = []
    for path in sorted(migrations_path.glob("*.sql")):
        prefix = path.stem.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        migrations.append((int(prefix), path))
    return migrations


def ensure_latest(conn: sqlite3.Connection, migrations_path: str | Path) -> None:
    """Apply all pending migrations in version order, safely and idempotently."""
    mpath = Path(migrations_path)
    current_version = get_schema_version(conn)

    for version, path in _discover_migrations(mpath):
        if version <= current_version:
            continue

        sql = path.read_text(encoding="utf-8")
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{sql}\n"
            f"UPDATE schema_version SET version = {version};\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
            current_version = version
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

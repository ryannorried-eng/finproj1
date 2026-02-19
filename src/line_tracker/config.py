"""Centralized configuration for Line Tracker.

Reads settings from Streamlit secrets (preferred on Cloud) then falls
back to environment variables.  Provides clear error messages when
required values are missing in a deployed environment.
"""

from __future__ import annotations

import os


def _read_secret(key: str, default: str | None = None) -> str | None:
    """Read *key* from ``st.secrets`` (if available) or ``os.environ``."""
    try:
        import streamlit as st

        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        # st.secrets unavailable (e.g. running outside Streamlit or no
        # secrets.toml configured).  Fall through to env.
        pass
    return os.environ.get(key, default)


def get_api_key() -> str:
    """Return the Odds API key or raise with a helpful message."""
    key = _read_secret("ODDS_API_KEY")
    if key:
        return key
    raise ValueError(
        "ODDS_API_KEY not found.  "
        "On Streamlit Cloud add it under Settings > Secrets.  "
        "Locally, export ODDS_API_KEY=<your-key> or add it to "
        ".streamlit/secrets.toml."
    )


def get_database_url() -> str | None:
    """Return DATABASE_URL if a persistent DB is configured, else None."""
    return _read_secret("DATABASE_URL")


def get_display_timezone() -> str:
    """Return the IANA timezone to use for display (default Central)."""
    return _read_secret("DISPLAY_TZ", "America/Chicago") or "America/Chicago"


def has_persistent_db() -> bool:
    """True when a DATABASE_URL or local SQLite file is accessible."""
    if get_database_url():
        return True
    # Check for the default SQLite path
    from line_tracker.storage import DEFAULT_DB_PATH

    return DEFAULT_DB_PATH.exists()


def data_mode() -> str:
    """Return ``'db'`` when a persistent database is available, else ``'live'``."""
    return "db" if has_persistent_db() else "live"

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


def get_db_path() -> str | None:
    """Return DB_PATH if a custom SQLite path is configured, else None.

    Set ``DB_PATH`` in Streamlit secrets or as an environment variable to
    point the app at a persistent volume path (e.g. Railway, Render, Fly.io).
    When unset, LineStore defaults to ~/.line_tracker/lines.db.
    """
    return _read_secret("DB_PATH")


def get_display_timezone() -> str:
    """Return the IANA timezone to use for display (default Central)."""
    return _read_secret("DISPLAY_TZ", "America/Chicago") or "America/Chicago"


def has_persistent_db() -> bool:
    """True when a persistent SQLite file is accessible."""
    configured = get_db_path()
    if configured:
        from pathlib import Path

        return Path(configured).exists()
    from line_tracker.storage import DEFAULT_DB_PATH

    return DEFAULT_DB_PATH.exists()


def data_mode() -> str:
    """Return ``'db'`` when a persistent database is available, else ``'live'``."""
    return "db" if has_persistent_db() else "live"


# ── Quantile-tier configuration ─────────────────────────────────────


def get_tier_method() -> str:
    """Return the tiering method (default ``"quantile"``)."""
    return _read_secret("TIER_METHOD", "quantile") or "quantile"


def get_tier1_quantile() -> float:
    """Top quantile share for Tier 1 (default 0.10 = top 10%)."""
    raw = _read_secret("TIER1_Q", "0.10")
    return float(raw) if raw else 0.10


def get_tier2_quantile() -> float:
    """Cumulative quantile for Tier 2 cutoff (default 0.35 = top 35%).

    Tier 2 spans from the Tier 1 cutoff down to this cumulative share.
    """
    raw = _read_secret("TIER2_Q", "0.35")
    return float(raw) if raw else 0.35


def get_tier_min_candidates() -> int:
    """Minimum candidate count for stable quantile tiers (default 10)."""
    raw = _read_secret("TIER_MIN_CANDIDATES", "10")
    return int(raw) if raw else 10

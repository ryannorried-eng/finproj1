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


# ── Market weighting configuration ────────────────────────────────


def get_market_weight_spread() -> float:
    """Market weight for spreads (default 1.00)."""
    raw = _read_secret("MARKET_W_SPREAD", "1.00")
    return float(raw) if raw else 1.00


def get_market_weight_total() -> float:
    """Market weight for totals (default 0.95)."""
    raw = _read_secret("MARKET_W_TOTAL", "0.95")
    return float(raw) if raw else 0.95


def get_market_weight_ml_fav() -> float:
    """Market weight for moneyline favorites (default 0.90)."""
    raw = _read_secret("MARKET_W_ML_FAV", "0.90")
    return float(raw) if raw else 0.90


def get_market_weight_ml_dog() -> float:
    """Market weight for moneyline underdogs (default 0.75)."""
    raw = _read_secret("MARKET_W_ML_DOG", "0.75")
    return float(raw) if raw else 0.75


def get_market_weight_longshot() -> float:
    """Longshot penalty for ML, consensus_prob < 0.20 (default 0.90)."""
    raw = _read_secret("MARKET_W_LONGSHOT", "0.90")
    return float(raw) if raw else 0.90


# ── Kelly effective multiplier configuration ──────────────────────


def get_kelly_mult_high() -> float:
    """Kelly effective multiplier for High confidence_label (default 1.00)."""
    raw = _read_secret("KELLY_MULT_HIGH", "1.00")
    return float(raw) if raw else 1.00


def get_kelly_mult_med() -> float:
    """Kelly effective multiplier for Medium confidence_label (default 0.70)."""
    raw = _read_secret("KELLY_MULT_MED", "0.70")
    return float(raw) if raw else 0.70


def get_kelly_mult_low() -> float:
    """Kelly effective multiplier for Low confidence_label (default 0.40)."""
    raw = _read_secret("KELLY_MULT_LOW", "0.40")
    return float(raw) if raw else 0.40


# ── Pro Hybrid confidence-weighted market weighting ───────────────


def get_pro_hybrid_market_conf() -> bool:
    """True when PRO_HYBRID_MARKET_CONF is enabled (default off)."""
    raw = _read_secret("PRO_HYBRID_MARKET_CONF", "0")
    return str(raw).strip() in ("1", "true", "True", "yes")


def get_conf_w_high() -> float:
    """Confidence weight factor for High label (default 1.15)."""
    raw = _read_secret("CONF_W_HIGH", "1.15")
    return float(raw) if raw else 1.15


def get_conf_w_med() -> float:
    """Confidence weight factor for Medium label (default 1.00)."""
    raw = _read_secret("CONF_W_MED", "1.00")
    return float(raw) if raw else 1.00


def get_conf_w_low() -> float:
    """Confidence weight factor for Low label (default 0.85)."""
    raw = _read_secret("CONF_W_LOW", "0.85")
    return float(raw) if raw else 0.85


def get_conf_w_min() -> float:
    """Minimum clamped effective market weight (default 0.50)."""
    raw = _read_secret("CONF_W_MIN", "0.50")
    return float(raw) if raw else 0.50


def get_conf_w_max() -> float:
    """Maximum clamped effective market weight (default 1.10)."""
    raw = _read_secret("CONF_W_MAX", "1.10")
    return float(raw) if raw else 1.10


# ── Pick-pruning gate thresholds ──────────────────────────────────


def get_prune_min_edge_z() -> float:
    """Minimum edge-z to survive pruning (default 1.15).

    Lowered from 1.75 to 1.15 to align with TIER3_MIN_EDGE_Z in slate.py,
    allowing Tier 3 candidates to survive pruning.  Other gates (quality,
    books, hold, ev_shrunk) still apply.
    """
    raw = _read_secret("PRUNE_MIN_EDGE_Z", "1.15")
    return float(raw) if raw else 1.15


def get_prune_min_ev_shrunk() -> float:
    """Minimum edge_ev_shrunk; entries must be strictly above this (default 0)."""
    raw = _read_secret("PRUNE_MIN_EV_SHRUNK", "0")
    return float(raw) if raw else 0.0


def get_prune_min_quality() -> int:
    """Minimum quality_score to survive pruning (default 65)."""
    raw = _read_secret("PRUNE_MIN_QUALITY", "65")
    return int(raw) if raw else 65


def get_prune_min_books() -> int:
    """Minimum books_used to survive pruning (default 4).

    Aligned with Tier 2 acceptance threshold (books_used >= 4) so that
    Tier 2 entries are not silently killed by a stricter prune gate.
    """
    raw = _read_secret("PRUNE_MIN_BOOKS", "4")
    return int(raw) if raw else 4


def get_prune_max_hold() -> float:
    """Maximum market_hold_median to survive pruning (default 7.0)."""
    raw = _read_secret("PRUNE_MAX_HOLD", "7.0")
    return float(raw) if raw else 7.0


def get_prune_allowed_tiers() -> frozenset[str]:
    """Allowed tiers for pruning (default tier1a,tier1b,tier2,tier3)."""
    raw = _read_secret("PRUNE_ALLOWED_TIERS", "tier1a,tier1b,tier2,tier3")
    if not raw:
        return frozenset({"tier1a", "tier1b", "tier2", "tier3"})
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def get_prune_allowed_alpha_labels() -> frozenset[str]:
    """Allowed alpha labels for pruning (default Strong,Neutral,Weak)."""
    raw = _read_secret("PRUNE_ALLOWED_ALPHA_LABELS", "Strong,Neutral,Weak")
    if not raw:
        return frozenset({"Strong", "Neutral", "Weak"})
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


# ── Slate markets configuration ─────────────────────────────────


def get_slate_markets() -> list[str]:
    """Return list of markets to include in slates (default moneyline,spread,total)."""
    raw = _read_secret("SLATE_MARKETS", "moneyline,spread,total")
    if not raw:
        return ["moneyline", "spread", "total"]
    return [m.strip().lower() for m in raw.split(",") if m.strip()]


# ── Odds API book / region configuration ─────────────────────────


def get_books_csv() -> str:
    """Return comma-separated bookmaker keys for Odds API requests."""
    default = (
        "fanduel,draftkings,betmgm,pointsbetus,"
        "caesars,barstool,betrivers,unibet_us"
    )
    return _read_secret("ODDS_BOOKS", default) or default


def get_regions() -> str:
    """Return regions parameter for Odds API requests (default 'us')."""
    return _read_secret("ODDS_REGIONS", "us") or "us"


# ── Edge-z sigma floor ──────────────────────────────────────────


def get_edge_z_sigma_min() -> float:
    """Minimum ev_sigma for edge_z computation to prevent compression (default 0.02)."""
    raw = _read_secret("EDGE_Z_SIGMA_MIN", "0.02")
    return float(raw) if raw else 0.02


# ── Debug / observability configuration ──────────────────────────


def get_prune_debug_mode() -> bool:
    """True when PRUNE_DEBUG_MODE is enabled (default off)."""
    raw = _read_secret("PRUNE_DEBUG_MODE", "0")
    return str(raw).strip() in ("1", "true", "True", "yes")


def get_slate_max_per_event() -> int:
    """Maximum recommendations per event in slate (default 1)."""
    raw = _read_secret("SLATE_MAX_PER_EVENT", "1")
    return int(raw) if raw else 1


def get_debug_prune_profile() -> dict:
    """Return effective pruning thresholds for debug mode.

    When PRUNE_DEBUG_MODE is enabled, relaxes tier and alpha gates and
    lowers edge_z / ev_shrunk floors so that Tier 3 candidates survive
    pruning and the next real bottleneck becomes visible.

    Normal-mode callers should NOT use this — it is only for diagnostics.
    """
    base_edge_z = get_prune_min_edge_z()
    base_ev_shrunk = get_prune_min_ev_shrunk()

    return {
        "allowed_tiers": frozenset({"tier1a", "tier1b", "tier2", "tier3"}),
        "allowed_alpha": frozenset({"Strong", "Neutral", "Weak", ""}),
        "min_edge_z": min(base_edge_z, 0.50),
        "min_ev_shrunk": min(base_ev_shrunk, 0.0) if base_ev_shrunk > 0 else base_ev_shrunk,
        "min_quality": get_prune_min_quality(),
        "min_books": get_prune_min_books(),
        "max_hold": get_prune_max_hold(),
    }

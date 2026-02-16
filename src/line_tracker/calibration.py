"""Tier Calibration – derive tier thresholds from historical CLV data."""

from __future__ import annotations

import json
from itertools import product

import pandas as pd

from line_tracker.performance import build_clv_dataframe

# ── Default tier targets ────────────────────────────────────────────
_TIER_TARGETS: dict[str, dict] = {
    "tier1a": {
        "beat_pct_min": 55.0,
        "avg_clv_min": 0.002,
        "n_min": 50,
        "legs_day_lo": 1.0,
        "legs_day_hi": 5.0,
    },
    "tier1b": {
        "beat_pct_min": 53.0,
        "avg_clv_min": 0.001,
        "n_min": 100,
        "legs_day_lo": 3.0,
        "legs_day_hi": 12.0,
    },
    "tier2": {
        "beat_pct_min": 51.0,
        "avg_clv_min": 0.0005,
        "n_min": 200,
        "legs_day_lo": 5.0,
        "legs_day_hi": 25.0,
    },
}

# ── Grid candidates ─────────────────────────────────────────────────
_EDGE_EV_100_CANDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
_EDGE_Z_CANDS = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
_HOLD_MAX_CANDS = [6.0, 6.5, 7.0, 7.5, 8.0]
_BOOKS_MIN_CANDS = [4, 5, 6, 7]

# ── Conservative defaults (fallback) ───────────────────────────────
_DEFAULTS: dict[str, dict] = {
    "tier1a": {
        "edge_ev_100": 2.0,
        "edge_z": 2.0,
        "hold_max": 6.0,
        "books_min": 6,
    },
    "tier1b": {
        "edge_ev_100": 1.0,
        "edge_z": 1.0,
        "hold_max": 7.5,
        "books_min": 5,
    },
    "tier2": {
        "edge_ev_100": 0.5,
        "edge_z": 0.5,
        "hold_max": 8.0,
        "books_min": 4,
    },
}


# ── Loading helpers ─────────────────────────────────────────────────


def load_clv_training_df(
    store,
    *,
    sport: str | None = None,
    market: str | None = None,
) -> pd.DataFrame:
    """Build a training DataFrame from stored CLV rows.

    Returns a DataFrame with columns needed for grid search:
        edge_ev_100, edge_z, hold, books_used, beat, clv, date.
    """
    rows = store.get_all_clv()
    if not rows:
        return pd.DataFrame()

    df = build_clv_dataframe(rows)
    if df.empty:
        return pd.DataFrame()

    # Filter by sport / market if requested
    if sport and "sport" in df.columns:
        df = df[df["sport"] == sport]
    if market and "market" in df.columns:
        df = df[df["market"] == market]

    # Require pick-time metadata
    required = {"edge_pct_at_pick", "edge_z_at_pick", "clv_prob"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Build training columns
    out = pd.DataFrame()
    out["edge_ev_100"] = pd.to_numeric(
        df["edge_pct_at_pick"], errors="coerce"
    ).fillna(0.0)
    out["edge_z"] = pd.to_numeric(
        df["edge_z_at_pick"], errors="coerce"
    ).fillna(0.0)
    out["hold"] = pd.to_numeric(
        df.get("market_hold_median_at_pick", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    out["books_used"] = pd.to_numeric(
        df.get("books_used_at_pick", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).astype(int)
    out["beat"] = (df["clv_prob"] > 0).astype(int)
    out["clv"] = df["clv_prob"].fillna(0.0)

    if "closed_at" in df.columns:
        out["date"] = pd.to_datetime(
            df["closed_at"], errors="coerce"
        ).dt.date

    return out.dropna(subset=["edge_ev_100", "edge_z"]).reset_index(
        drop=True
    )


# ── Grid search ─────────────────────────────────────────────────────


def grid_search_thresholds(
    df: pd.DataFrame,
    tier_name: str,
    constraints: dict | None = None,
) -> dict:
    """Search for best (edge_ev_100, edge_z, hold_max, books_min) combo.

    Returns dict with keys: edge_ev_100, edge_z, hold_max, books_min,
    n, beat_rate, avg_clv, median_clv, score, legs_per_day,
    fallback_used (bool).
    """
    cons = constraints or _TIER_TARGETS.get(tier_name, _TIER_TARGETS["tier2"])
    n_min = cons.get("n_min", 50)
    legs_day_lo = cons.get("legs_day_lo", 1.0)
    legs_day_hi = cons.get("legs_day_hi", 25.0)

    if df.empty:
        default = _DEFAULTS[tier_name].copy()
        default.update({
            "n": 0, "beat_rate": 0.0, "avg_clv": 0.0,
            "median_clv": 0.0, "score": 0.0,
            "legs_per_day": 0.0, "fallback_used": True,
        })
        return default

    # Compute date range for legs/day
    num_days = 1.0
    if "date" in df.columns and len(df) > 0:
        dates = df["date"].dropna()
        if len(dates) > 1:
            d_min, d_max = dates.min(), dates.max()
            span = (d_max - d_min).days if hasattr(d_max - d_min, "days") else 1
            num_days = max(span, 1)

    best_score = float("-inf")
    best_result: dict | None = None

    for ev_100, ez, hmax, bmin in product(
        _EDGE_EV_100_CANDS, _EDGE_Z_CANDS, _HOLD_MAX_CANDS, _BOOKS_MIN_CANDS
    ):
        mask = (
            (df["edge_ev_100"] >= ev_100)
            & (df["edge_z"] >= ez)
            & (df["hold"] <= hmax)
            & (df["books_used"] >= bmin)
        )
        subset = df[mask]
        n = len(subset)
        if n < n_min:
            continue

        beat_rate = 100.0 * subset["beat"].mean()
        avg_clv = float(subset["clv"].mean())
        legs_day = n / num_days

        # Play-volume guard
        if legs_day < legs_day_lo or legs_day > legs_day_hi:
            continue

        score = 100.0 * (beat_rate / 100.0 - 0.50) + 500.0 * avg_clv

        if score > best_score:
            best_score = score
            best_result = {
                "edge_ev_100": ev_100,
                "edge_z": ez,
                "hold_max": hmax,
                "books_min": bmin,
                "n": n,
                "beat_rate": round(beat_rate, 1),
                "avg_clv": round(avg_clv, 6),
                "median_clv": round(float(subset["clv"].median()), 6),
                "score": round(score, 4),
                "legs_per_day": round(legs_day, 2),
                "fallback_used": False,
            }

    if best_result is not None:
        return best_result

    # Fallback to conservative defaults
    default = _DEFAULTS[tier_name].copy()
    default.update({
        "n": 0,
        "beat_rate": 0.0,
        "avg_clv": 0.0,
        "median_clv": 0.0,
        "score": 0.0,
        "legs_per_day": 0.0,
        "fallback_used": True,
    })
    return default


# ── Calibrate all tiers ─────────────────────────────────────────────


def _enforce_monotonicity(tiers: dict) -> dict:
    """Ensure tier1a >= tier1b >= tier2 for edge/z/books and <= for hold."""
    t1a = tiers["tier1a"]
    t1b = tiers["tier1b"]
    t2 = tiers["tier2"]

    # edge_ev_100: 1a >= 1b >= 2
    if t1b["edge_ev_100"] > t1a["edge_ev_100"]:
        t1b["edge_ev_100"] = t1a["edge_ev_100"]
    if t2["edge_ev_100"] > t1b["edge_ev_100"]:
        t2["edge_ev_100"] = t1b["edge_ev_100"]

    # edge_z: 1a >= 1b >= 2
    if t1b["edge_z"] > t1a["edge_z"]:
        t1b["edge_z"] = t1a["edge_z"]
    if t2["edge_z"] > t1b["edge_z"]:
        t2["edge_z"] = t1b["edge_z"]

    # books_min: 1a >= 1b >= 2
    if t1b["books_min"] > t1a["books_min"]:
        t1b["books_min"] = t1a["books_min"]
    if t2["books_min"] > t1b["books_min"]:
        t2["books_min"] = t1b["books_min"]

    # hold_max: 1a <= 1b <= 2
    if t1b["hold_max"] < t1a["hold_max"]:
        t1b["hold_max"] = t1a["hold_max"]
    if t2["hold_max"] < t1b["hold_max"]:
        t2["hold_max"] = t1b["hold_max"]

    return tiers


def calibrate_thresholds(df: pd.DataFrame) -> dict:
    """Run grid search for all three tiers and enforce monotonicity.

    Returns dict with keys: tier1a, tier1b, tier2 (each a threshold dict),
    plus training_rows, date_range, fallback_used.
    """
    if df.empty:
        result = {
            tier: {**_DEFAULTS[tier], "fallback_used": True}
            for tier in ("tier1a", "tier1b", "tier2")
        }
        result["training_rows"] = 0
        result["date_range"] = None
        result["fallback_used"] = True
        return result

    t1a = grid_search_thresholds(df, "tier1a")
    t1b = grid_search_thresholds(df, "tier1b")
    t2 = grid_search_thresholds(df, "tier2")

    tiers = {"tier1a": t1a, "tier1b": t1b, "tier2": t2}
    tiers = _enforce_monotonicity(tiers)

    date_range = None
    if "date" in df.columns:
        dates = df["date"].dropna()
        if len(dates) > 0:
            date_range = f"{dates.min()} to {dates.max()}"

    return {
        **tiers,
        "training_rows": len(df),
        "date_range": date_range,
        "fallback_used": all(
            t.get("fallback_used", False) for t in (t1a, t1b, t2)
        ),
    }


# ── Report formatting ───────────────────────────────────────────────


def format_calibration_report(result: dict) -> str:
    """Human-readable calibration summary."""
    lines = ["Tier Calibration Report", "=" * 40]

    lines.append(f"Training rows: {result.get('training_rows', 0)}")
    if result.get("date_range"):
        lines.append(f"Date range: {result['date_range']}")
    if result.get("fallback_used"):
        lines.append("WARNING: Using fallback defaults (insufficient data)")

    for tier in ("tier1a", "tier1b", "tier2"):
        t = result.get(tier, {})
        lines.append("")
        lines.append(f"--- {tier.upper()} ---")
        lines.append(f"  edge_ev_100 >= {t.get('edge_ev_100', '?')}")
        lines.append(f"  edge_z      >= {t.get('edge_z', '?')}")
        lines.append(f"  hold_max    <= {t.get('hold_max', '?')}%")
        lines.append(f"  books_min   >= {t.get('books_min', '?')}")
        if "n" in t:
            lines.append(f"  training n  =  {t['n']}")
            lines.append(f"  beat_rate   =  {t.get('beat_rate', 0)}%")
            lines.append(f"  avg_clv     =  {t.get('avg_clv', 0):.4f}")
            lines.append(f"  legs/day    =  {t.get('legs_per_day', 0)}")
        if t.get("fallback_used"):
            lines.append("  (fallback defaults)")

    return "\n".join(lines)


# ── JSON serialisation helpers ──────────────────────────────────────


def calibration_to_json(result: dict) -> str:
    """Serialise calibration result to JSON string."""
    return json.dumps(result)


def calibration_from_json(json_str: str) -> dict:
    """Deserialise calibration JSON back to dict."""
    return json.loads(json_str)

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


# ── Rolling CLV statistics ──────────────────────────────────────────
_ROLLING_WINDOW_DAYS = 30  # rolling window for CLV mean / volatility
_ROLLING_VOL_PENALTY_MULT = 200.0  # score penalty per unit of CLV vol
_ROLLING_MEAN_BONUS_MULT = 300.0  # score bonus per unit of rolling CLV mean


def compute_rolling_clv_stats(
    df: pd.DataFrame,
    window_days: int = _ROLLING_WINDOW_DAYS,
) -> dict:
    """Compute rolling CLV mean and volatility from the training DataFrame.

    Uses the most recent *window_days* of data (by ``date`` column) to
    compute rolling statistics that inform threshold adjustment.

    Returns
    -------
    dict with keys:
        rolling_clv_mean, rolling_clv_std, rolling_beat_rate,
        rolling_n, window_days, has_rolling_data.
    """
    if df.empty or "date" not in df.columns or "clv" not in df.columns:
        return {
            "rolling_clv_mean": 0.0,
            "rolling_clv_std": 0.0,
            "rolling_beat_rate": 0.0,
            "rolling_n": 0,
            "window_days": window_days,
            "has_rolling_data": False,
        }

    dates = df["date"].dropna()
    if dates.empty:
        return {
            "rolling_clv_mean": 0.0,
            "rolling_clv_std": 0.0,
            "rolling_beat_rate": 0.0,
            "rolling_n": 0,
            "window_days": window_days,
            "has_rolling_data": False,
        }

    from datetime import timedelta

    max_date = dates.max()
    cutoff = max_date - timedelta(days=window_days)
    recent = df[df["date"] > cutoff]

    if len(recent) < 10:
        return {
            "rolling_clv_mean": 0.0,
            "rolling_clv_std": 0.0,
            "rolling_beat_rate": 0.0,
            "rolling_n": len(recent),
            "window_days": window_days,
            "has_rolling_data": False,
        }

    clv_vals = recent["clv"].dropna()
    beat_vals = recent["beat"] if "beat" in recent.columns else pd.Series(dtype=float)

    return {
        "rolling_clv_mean": round(float(clv_vals.mean()), 6),
        "rolling_clv_std": (
            round(float(clv_vals.std()), 6) if len(clv_vals) > 1 else 0.0
        ),
        "rolling_beat_rate": (
            round(float(beat_vals.mean() * 100), 1)
            if len(beat_vals) > 0 else 0.0
        ),
        "rolling_n": len(recent),
        "window_days": window_days,
        "has_rolling_data": True,
    }


def _rolling_score_adjustment(rolling_stats: dict) -> float:
    """Compute a score adjustment based on rolling CLV statistics.

    Positive adjustment = recent performance is good (relax thresholds).
    Negative adjustment = recent performance is poor (tighten thresholds).

    The adjustment modifies the grid search scoring function to prefer
    combinations that account for recent CLV trends.
    """
    if not rolling_stats.get("has_rolling_data"):
        return 0.0

    mean_adj = _ROLLING_MEAN_BONUS_MULT * rolling_stats.get("rolling_clv_mean", 0.0)
    vol_penalty = _ROLLING_VOL_PENALTY_MULT * rolling_stats.get("rolling_clv_std", 0.0)

    return mean_adj - vol_penalty


# ── Grid search ─────────────────────────────────────────────────────


def grid_search_thresholds(
    df: pd.DataFrame,
    tier_name: str,
    constraints: dict | None = None,
    rolling_stats: dict | None = None,
) -> dict:
    """Search for best (edge_ev_100, edge_z, hold_max, books_min) combo.

    When *rolling_stats* is provided (from ``compute_rolling_clv_stats``),
    the scoring function is adjusted to account for recent CLV trends:
    positive rolling CLV mean adds a bonus, high rolling volatility adds
    a penalty.  This causes Auto mode to tighten thresholds when recent
    performance is declining.

    Returns dict with keys: edge_ev_100, edge_z, hold_max, books_min,
    n, beat_rate, avg_clv, median_clv, score, legs_per_day,
    fallback_used (bool), and optionally rolling_clv_mean,
    rolling_clv_std.
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

    # Rolling CLV adjustment
    roll_adj = _rolling_score_adjustment(rolling_stats or {})

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

        score = 100.0 * (beat_rate / 100.0 - 0.50) + 500.0 * avg_clv + roll_adj

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
        # Attach rolling stats metadata if available
        if rolling_stats and rolling_stats.get("has_rolling_data"):
            best_result["rolling_clv_mean"] = rolling_stats["rolling_clv_mean"]
            best_result["rolling_clv_std"] = rolling_stats["rolling_clv_std"]
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


def calibrate_thresholds(
    df: pd.DataFrame,
    *,
    use_rolling: bool = True,
) -> dict:
    """Run grid search for all three tiers and enforce monotonicity.

    When *use_rolling* is True (default), rolling CLV statistics from
    the most recent window are computed and used to adjust the scoring
    function.  This causes the auto-calibration to adapt to recent
    performance trends without requiring a schema change.

    Returns dict with keys: tier1a, tier1b, tier2 (each a threshold dict),
    plus training_rows, date_range, fallback_used, and rolling_clv.
    """
    if df.empty:
        result = {
            tier: {**_DEFAULTS[tier], "fallback_used": True}
            for tier in ("tier1a", "tier1b", "tier2")
        }
        result["training_rows"] = 0
        result["date_range"] = None
        result["fallback_used"] = True
        result["rolling_clv"] = compute_rolling_clv_stats(df)
        return result

    rolling_stats = compute_rolling_clv_stats(df) if use_rolling else {}

    t1a = grid_search_thresholds(df, "tier1a", rolling_stats=rolling_stats)
    t1b = grid_search_thresholds(df, "tier1b", rolling_stats=rolling_stats)
    t2 = grid_search_thresholds(df, "tier2", rolling_stats=rolling_stats)

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
        "rolling_clv": rolling_stats,
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

    # Rolling CLV stats
    rolling = result.get("rolling_clv", {})
    if rolling.get("has_rolling_data"):
        lines.append("")
        lines.append("--- ROLLING CLV ---")
        lines.append(f"  window:     {rolling.get('window_days', 30)} days")
        lines.append(f"  n:          {rolling.get('rolling_n', 0)}")
        lines.append(f"  CLV mean:   {rolling.get('rolling_clv_mean', 0):.6f}")
        lines.append(f"  CLV std:    {rolling.get('rolling_clv_std', 0):.6f}")
        lines.append(f"  beat rate:  {rolling.get('rolling_beat_rate', 0):.1f}%")

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
        if t.get("rolling_clv_mean") is not None:
            lines.append(f"  roll_mean   =  {t.get('rolling_clv_mean', 0):.6f}")
            lines.append(f"  roll_std    =  {t.get('rolling_clv_std', 0):.6f}")
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

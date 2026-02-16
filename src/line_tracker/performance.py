"""CLV-driven Performance analytics for the dashboard.

Provides:
  - load_clv_df:  Load CLV rows into a pandas DataFrame with filters.
  - compute_kpis: Summary KPIs (beat-close %, avg/median CLV, sample size).
  - compute_breakdown_tables: Group-by breakdowns (tier/confidence/market/
    sport/book).
  - compute_trends: Rolling time-series of CLV metrics.
  - calibration_suggestions: Actionable messages based on CLV data.
"""

from __future__ import annotations

import pandas as pd

from line_tracker.storage import LineStore

# Minimum sample size before making strong calibration claims.
MIN_SAMPLE_STRONG = 30
MIN_SAMPLE_TOTAL = 100


# ---------------------------------------------------------------------------
# 1. Load CLV rows into DataFrame
# ---------------------------------------------------------------------------

def load_clv_df(
    store: LineStore,
    *,
    start: str | None = None,
    end: str | None = None,
    sport: str | None = None,
    market: str | None = None,
    book: str | None = None,
    confidence: str | None = None,
    tier: str | None = None,
    include_estimated: bool = False,
) -> pd.DataFrame:
    """Load CLV analytics rows from the store as a DataFrame.

    Rows with no exec close data are excluded unless the caller sets
    *include_estimated*.
    """
    rows = store.get_clv_rows(
        start=start,
        end=end,
        sport=sport,
        market=market,
        book=book,
        confidence=confidence,
        tier=tier,
        include_estimated=include_estimated,
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Parse settled_at to datetime for trend analysis
    if "settled_at" in df.columns:
        df["settled_dt"] = pd.to_datetime(
            df["settled_at"], errors="coerce", utc=True,
        )

    return df


# ---------------------------------------------------------------------------
# 2. KPIs
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute summary KPIs from a CLV DataFrame.

    Returns a dict with keys:
      - sample_size
      - pct_beat_close_exec
      - avg_clv_prob_pts_exec
      - median_clv_prob_pts_exec
      - pct_beat_close_market
      - avg_clv_prob_pts_market
      - median_clv_prob_pts_market
    """
    empty = {
        "sample_size": 0,
        "pct_beat_close_exec": None,
        "avg_clv_prob_pts_exec": None,
        "median_clv_prob_pts_exec": None,
        "pct_beat_close_market": None,
        "avg_clv_prob_pts_market": None,
        "median_clv_prob_pts_market": None,
    }
    if df.empty:
        return empty

    # Exec CLV
    exec_valid = df["exec_clv_prob"].dropna()
    n = len(exec_valid)
    if n == 0:
        return {**empty, "sample_size": len(df)}

    beat_exec = df["beat_close_exec"].dropna()
    pct_beat_exec = (
        beat_exec.mean() * 100 if len(beat_exec) > 0 else None
    )

    # Market CLV
    mkt_valid = df["market_clv_prob"].dropna()
    beat_mkt = df["beat_close_market"].dropna()
    pct_beat_mkt = (
        beat_mkt.mean() * 100 if len(beat_mkt) > 0 else None
    )

    return {
        "sample_size": n,
        "pct_beat_close_exec": (
            round(pct_beat_exec, 1) if pct_beat_exec is not None
            else None
        ),
        "avg_clv_prob_pts_exec": round(exec_valid.mean() * 100, 2),
        "median_clv_prob_pts_exec": round(
            exec_valid.median() * 100, 2,
        ),
        "pct_beat_close_market": (
            round(pct_beat_mkt, 1) if pct_beat_mkt is not None
            else None
        ),
        "avg_clv_prob_pts_market": (
            round(mkt_valid.mean() * 100, 2)
            if len(mkt_valid) > 0 else None
        ),
        "median_clv_prob_pts_market": (
            round(mkt_valid.median() * 100, 2)
            if len(mkt_valid) > 0 else None
        ),
    }


# ---------------------------------------------------------------------------
# 3. Breakdown tables
# ---------------------------------------------------------------------------

_BREAKDOWN_COLS = [
    "count", "pct_beat_close", "avg_clv_prob_pts",
    "median_clv_prob_pts", "avg_hold", "avg_sigma",
    "avg_edge_pct", "avg_edge_z",
]


def _group_stats(grp: pd.DataFrame) -> dict:
    """Compute stats for a single group."""
    beat = grp["beat_close_exec"].dropna()
    clv = grp["exec_clv_prob"].dropna()
    return {
        "count": len(grp),
        "pct_beat_close": (
            round(beat.mean() * 100, 1) if len(beat) > 0 else None
        ),
        "avg_clv_prob_pts": (
            round(clv.mean() * 100, 2) if len(clv) > 0 else None
        ),
        "median_clv_prob_pts": (
            round(clv.median() * 100, 2) if len(clv) > 0 else None
        ),
        "avg_hold": (
            round(grp["market_hold_median"].dropna().mean() * 100, 2)
            if grp["market_hold_median"].notna().any() else None
        ),
        "avg_sigma": (
            round(
                grp["market_volatility_sigma"].dropna().mean() * 100,
                2,
            )
            if grp["market_volatility_sigma"].notna().any() else None
        ),
        "avg_edge_pct": (
            round(grp["edge_pct"].dropna().mean() * 100, 2)
            if grp["edge_pct"].notna().any() else None
        ),
        "avg_edge_z": (
            round(grp["edge_z"].dropna().mean(), 2)
            if grp["edge_z"].notna().any() else None
        ),
    }


def compute_breakdown_tables(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute breakdown tables grouped by tier, confidence, market,
    sport, and book.

    Returns a dict of DataFrames keyed by group name.
    """
    result: dict[str, pd.DataFrame] = {}
    if df.empty:
        for key in (
            "by_tier", "by_confidence", "by_market",
            "by_sport", "by_book",
        ):
            result[key] = pd.DataFrame(columns=_BREAKDOWN_COLS)
        return result

    groupings = {
        "by_tier": "quality_tier",
        "by_confidence": "confidence",
        "by_market": "market",
        "by_sport": "sport",
        "by_book": "pick_sportsbook",
    }
    for key, col in groupings.items():
        if col not in df.columns or df[col].isna().all():
            result[key] = pd.DataFrame(columns=_BREAKDOWN_COLS)
            continue
        rows = []
        for name, grp in df.groupby(col, dropna=True):
            stats = _group_stats(grp)
            stats["group"] = name
            rows.append(stats)
        tbl = pd.DataFrame(rows)
        if not tbl.empty:
            tbl = tbl.set_index("group")
        result[key] = tbl

    return result


# ---------------------------------------------------------------------------
# 4. Rolling trends
# ---------------------------------------------------------------------------

def compute_trends(df: pd.DataFrame) -> dict:
    """Compute rolling daily trends for CLV metrics.

    Returns dict with:
      - daily: DataFrame with date index, pct_beat_close_exec,
        avg_clv_prob_pts_exec columns.
      - rolling_7: 7-day rolling average.
      - rolling_30: 30-day rolling average.
      - reason: str | None — explanation if data is insufficient.
    """
    empty = {
        "daily": pd.DataFrame(),
        "rolling_7": pd.DataFrame(),
        "rolling_30": pd.DataFrame(),
        "reason": None,
    }

    if df.empty or "settled_dt" not in df.columns:
        empty["reason"] = "No CLV data available."
        return empty

    valid = df.dropna(subset=["exec_clv_prob", "settled_dt"]).copy()
    if len(valid) < 3:
        empty["reason"] = (
            f"Only {len(valid)} closed leg(s) — need at least 3 "
            "for trend analysis."
        )
        return empty

    valid["date"] = valid["settled_dt"].dt.date

    daily_agg = valid.groupby("date").agg(
        pct_beat_close_exec=("beat_close_exec", "mean"),
        avg_clv_prob_pts_exec=("exec_clv_prob", "mean"),
    )
    daily_agg["pct_beat_close_exec"] *= 100
    daily_agg["avg_clv_prob_pts_exec"] *= 100
    daily_agg = daily_agg.sort_index()

    rolling_7 = daily_agg.rolling(7, min_periods=1).mean()
    rolling_30 = daily_agg.rolling(30, min_periods=1).mean()

    return {
        "daily": daily_agg,
        "rolling_7": rolling_7,
        "rolling_30": rolling_30,
        "reason": None,
    }


# ---------------------------------------------------------------------------
# 5. Calibration suggestions
# ---------------------------------------------------------------------------

def calibration_suggestions(
    df: pd.DataFrame,
    by_tier: pd.DataFrame,
    by_conf: pd.DataFrame,
) -> list[str]:
    """Generate 3–6 actionable calibration messages.

    Uses conservative thresholds and requires minimum sample sizes
    before making strong statements.
    """
    msgs: list[str] = []

    total = len(df) if not df.empty else 0
    if total < MIN_SAMPLE_TOTAL:
        msgs.append(
            f"Need more closed legs (target {MIN_SAMPLE_TOTAL}+) "
            "for reliable calibration. "
            f"Currently have {total}."
        )
        return msgs

    # --- Tier comparison ---
    if not by_tier.empty:
        _tier_suggestions(by_tier, msgs)

    # --- Confidence comparison ---
    if not by_conf.empty:
        _confidence_suggestions(by_conf, msgs)

    # --- Hold analysis ---
    if not df.empty and "market_hold_median" in df.columns:
        _hold_suggestions(df, msgs)

    # Ensure we return at least one message
    if not msgs:
        msgs.append(
            "CLV metrics look reasonable. Keep collecting data "
            "for more precise calibration."
        )

    return msgs[:6]


def _tier_suggestions(by_tier: pd.DataFrame, msgs: list[str]) -> None:
    """Add tier-based suggestions."""
    t1 = by_tier.loc["Tier1"] if "Tier1" in by_tier.index else None
    t2 = by_tier.loc["Tier2"] if "Tier2" in by_tier.index else None
    sa = (
        by_tier.loc["StayAway"]
        if "StayAway" in by_tier.index else None
    )

    if t1 is not None and t2 is not None:
        n1, n2 = t1["count"], t2["count"]
        if n1 >= MIN_SAMPLE_STRONG and n2 >= MIN_SAMPLE_STRONG:
            c1 = t1["avg_clv_prob_pts"]
            c2 = t2["avg_clv_prob_pts"]
            if (
                c1 is not None and c2 is not None
                and c1 <= c2
            ):
                msgs.append(
                    "Tier 1 does not outperform Tier 2 in avg CLV "
                    f"({c1:+.2f} vs {c2:+.2f} prob pts) "
                    "-- consider raising Tier 1 edge_z threshold."
                )
            elif c1 is not None and c2 is not None and c1 > c2:
                msgs.append(
                    f"Tier 1 outperforms Tier 2 ({c1:+.2f} vs "
                    f"{c2:+.2f} prob pts) -- tier system working."
                )

    if sa is not None and sa["count"] >= MIN_SAMPLE_STRONG:
        sa_clv = sa["avg_clv_prob_pts"]
        if sa_clv is not None and sa_clv > 0:
            msgs.append(
                "StayAway tier has positive CLV "
                f"({sa_clv:+.2f} prob pts) "
                "-- consider promoting some of these picks."
            )


def _confidence_suggestions(
    by_conf: pd.DataFrame, msgs: list[str],
) -> None:
    """Add confidence-based suggestions."""
    low = by_conf.loc["Low"] if "Low" in by_conf.index else None
    if low is not None and low["count"] >= MIN_SAMPLE_STRONG:
        low_clv = low["avg_clv_prob_pts"]
        if low_clv is not None and low_clv < 0:
            msgs.append(
                "Low-confidence picks have negative CLV "
                f"({low_clv:+.2f} prob pts) "
                "-- keep Low out of Tier 2."
            )


def _hold_suggestions(df: pd.DataFrame, msgs: list[str]) -> None:
    """Add hold-based suggestions."""
    hold_col = df["market_hold_median"].dropna()
    if len(hold_col) < MIN_SAMPLE_STRONG:
        return

    high_hold = df[df["market_hold_median"] > 0.05]
    low_hold = df[df["market_hold_median"] <= 0.05]
    if len(high_hold) >= MIN_SAMPLE_STRONG and len(low_hold) > 0:
        hh_clv = high_hold["exec_clv_prob"].dropna()
        lh_clv = low_hold["exec_clv_prob"].dropna()
        if len(hh_clv) > 0 and len(lh_clv) > 0:
            hh_avg = hh_clv.mean() * 100
            lh_avg = lh_clv.mean() * 100
            if hh_avg < lh_avg - 0.5:
                msgs.append(
                    f"High-hold markets underperform "
                    f"({hh_avg:+.2f} vs {lh_avg:+.2f} prob pts) "
                    "-- tighten hold cap."
                )

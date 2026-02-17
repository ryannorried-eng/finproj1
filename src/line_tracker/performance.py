"""CLV performance analytics for the Performance page."""

from __future__ import annotations

import pandas as pd

from line_tracker.core.clv import compute_clv_metrics


def build_clv_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert raw CLV rows from the database into an analytics DataFrame.

    Each row is expected to have at minimum:
        pick_odds_decimal, best_odds_close_decimal,
        consensus_prob_at_pick, consensus_prob_close, closed_at.

    Computed columns added:
        clv_decimal  – pick_dec - close_dec (positive = beat the close)
        clv_prob     – close_prob - pick_prob (positive = beat the close)
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Compute CLV metrics from the canonical helper.
    pick_prob = pd.to_numeric(df["consensus_prob_at_pick"], errors="coerce")
    close_prob_raw = pd.to_numeric(df["consensus_prob_close"], errors="coerce")
    close_prob = close_prob_raw.where(close_prob_raw.notna(), pick_prob)
    metrics = compute_clv_metrics(
        pick_dec=pd.to_numeric(df["pick_odds_decimal"], errors="coerce"),
        close_dec=pd.to_numeric(df["best_odds_close_decimal"], errors="coerce"),
        pick_prob=pick_prob,
        close_prob=close_prob,
    )
    df["clv_decimal"] = metrics["clv_decimal"]
    df["clv_prob"] = metrics["clv_prob"]

    # Parse closed_at to datetime
    if "closed_at" in df.columns:
        df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")

    return df


# ------------------------------------------------------------------
# Summary KPIs
# ------------------------------------------------------------------

def summary_kpis(df: pd.DataFrame) -> dict:
    """Compute top-level performance KPIs from a CLV DataFrame.

    Returns a dict with:
        total_legs, beating_close_pct, avg_clv_decimal, median_clv_decimal,
        avg_clv_prob, median_clv_prob.
    """
    if df.empty:
        return {
            "total_legs": 0,
            "beating_close_pct": 0.0,
            "avg_clv_decimal": 0.0,
            "median_clv_decimal": 0.0,
            "avg_clv_prob": 0.0,
            "median_clv_prob": 0.0,
        }

    total = len(df)
    # Canonical CLV sign convention:
    #   clv_prob > 0   => consensus moved toward your pick (beat close)
    #   clv_decimal > 0 => pick decimal better than close (beat close)
    beating = (df["clv_prob"] > 0).sum()

    return {
        "total_legs": total,
        "beating_close_pct": round(100.0 * beating / total, 1),
        "avg_clv_decimal": round(float(df["clv_decimal"].mean()), 4),
        "median_clv_decimal": round(float(df["clv_decimal"].median()), 4),
        "avg_clv_prob": round(float(df["clv_prob"].mean()), 4),
        "median_clv_prob": round(float(df["clv_prob"].median()), 4),
    }


# ------------------------------------------------------------------
# Group-by breakdowns
# ------------------------------------------------------------------

_GROUPBY_COLUMNS = [
    "confidence_at_pick",
    "quality_tier_at_pick",
    "market",
    "sport",
    "pick_sportsbook",
]


def groupby_breakdown(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate CLV stats grouped by *column*.

    Returns a DataFrame with columns:
        <column>, legs, beating_pct, avg_clv_decimal, avg_clv_prob.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=[column])
    if valid.empty:
        return pd.DataFrame()

    def _agg(group: pd.DataFrame) -> dict:
        n = len(group)
        return {
            "legs": n,
            "beating_pct": round(100.0 * (group["clv_prob"] > 0).sum() / n, 1),
            "avg_clv_decimal": round(float(group["clv_decimal"].mean()), 4),
            "avg_clv_prob": round(float(group["clv_prob"].mean()), 4),
        }

    records = []
    for key, grp in valid.groupby(column, sort=True):
        row = _agg(grp)
        row[column] = key
        records.append(row)

    result = pd.DataFrame(records)
    # Reorder so group column is first
    cols = [column] + [c for c in result.columns if c != column]
    return result[cols]


def all_breakdowns(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run groupby_breakdown for every standard grouping column.

    Returns a dict mapping column name → breakdown DataFrame.
    Only includes columns that actually exist in df.
    """
    out = {}
    for col in _GROUPBY_COLUMNS:
        tbl = groupby_breakdown(df, col)
        if not tbl.empty:
            out[col] = tbl
    return out


# ------------------------------------------------------------------
# Rolling time-series (30-day window)
# ------------------------------------------------------------------

def rolling_clv_series(
    df: pd.DataFrame, window_days: int = 30
) -> pd.DataFrame:
    """Compute a rolling mean of clv_prob over *window_days*.

    Returns a DataFrame with columns: date, rolling_clv_prob, cumulative_legs.
    Indexed by calendar date.
    """
    if df.empty or "closed_at" not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=["closed_at"]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["date"] = valid["closed_at"].dt.date

    daily = valid.groupby("date").agg(
        daily_clv_prob=("clv_prob", "mean"),
        legs=("clv_prob", "count"),
    ).sort_index()

    daily["rolling_clv_prob"] = (
        daily["daily_clv_prob"].rolling(window_days, min_periods=1).mean()
    )
    daily["cumulative_legs"] = daily["legs"].cumsum()

    return daily.reset_index()


# ------------------------------------------------------------------
# CLV distribution bins
# ------------------------------------------------------------------

_DEFAULT_BINS = [
    -0.10, -0.05, -0.03, -0.01,
    0.0,
    0.01, 0.03, 0.05, 0.10,
]


def clv_distribution(
    df: pd.DataFrame,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Bin clv_prob values into a distribution histogram.

    Returns a DataFrame with columns: bin_label, count, pct.
    """
    if df.empty:
        return pd.DataFrame()

    edges = bins if bins is not None else _DEFAULT_BINS
    # Add -inf and +inf at edges
    full_edges = [float("-inf")] + edges + [float("inf")]

    labels = []
    for i in range(len(full_edges) - 1):
        lo = full_edges[i]
        hi = full_edges[i + 1]
        if lo == float("-inf"):
            labels.append(f"< {hi:+.2f}")
        elif hi == float("inf"):
            labels.append(f">= {lo:+.2f}")
        else:
            labels.append(f"[{lo:+.2f}, {hi:+.2f})")

    series = pd.cut(
        df["clv_prob"],
        bins=full_edges,
        labels=labels,
        right=False,
    )
    counts = series.value_counts().reindex(labels, fill_value=0)
    total = counts.sum()
    result = pd.DataFrame({
        "bin_label": counts.index,
        "count": counts.values,
        "pct": (counts.values / total * 100).round(1) if total > 0 else 0.0,
    })
    return result


# ------------------------------------------------------------------
# Filtering helper
# ------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    sport: str | None = None,
    market: str | None = None,
    confidence: str | None = None,
    quality_tier: str | None = None,
) -> pd.DataFrame:
    """Apply optional filters to the CLV DataFrame."""
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    def _normalized_bound(value: str) -> pd.Timestamp:
        bound = pd.Timestamp(value)
        if "closed_at" in df.columns and isinstance(df["closed_at"].dtype, pd.DatetimeTZDtype):
            tz = df["closed_at"].dt.tz
            if bound.tzinfo is None:
                return bound.tz_localize(tz)
            return bound.tz_convert(tz)
        if bound.tzinfo is not None:
            return bound.tz_localize(None)
        return bound

    if date_start and "closed_at" in df.columns:
        mask &= df["closed_at"] >= _normalized_bound(date_start)
    if date_end and "closed_at" in df.columns:
        # End date is inclusive at the day level:
        # selecting YYYY-MM-DD should include all timestamps on that date.
        end_exclusive = _normalized_bound(date_end) + pd.Timedelta(days=1)
        mask &= df["closed_at"] < end_exclusive
    if sport and "sport" in df.columns:
        mask &= df["sport"] == sport
    if market and "market" in df.columns:
        mask &= df["market"] == market
    if confidence and "confidence_at_pick" in df.columns:
        mask &= df["confidence_at_pick"] == confidence
    if quality_tier and "quality_tier_at_pick" in df.columns:
        mask &= df["quality_tier_at_pick"] == quality_tier

    return df[mask]


def clv_color(clv_prob: float) -> str:
    """Return display color for CLV based on probability CLV semantics.

    Positive CLV probability means we beat the close (good = green),
    negative means we lost to the close (bad = red), and zero is neutral.
    """
    if clv_prob > 0:
        return "green"
    if clv_prob < 0:
        return "red"
    return "gray"


# ------------------------------------------------------------------
# Calibration: tier-proxy CLV comparison
# ------------------------------------------------------------------

# Thresholds matching the standard classify_rec Tier 1 / Tier 2 gates
_CAL_T1_EDGE = 3.0
_CAL_T2_EDGE = 1.5


def _tier_proxy(row: pd.Series) -> str:
    """Assign a proxy tier label based on pick-time metadata.

    Uses the same gate logic as ``classify_rec`` (standard mode) but
    applied to CLV-table columns which use ``_at_pick`` suffixes.
    """
    conf = row.get("confidence_at_pick", "")
    qt = row.get("quality_tier_at_pick", "")
    edge = row.get("edge_pct_at_pick") or 0.0

    if conf == "High" and qt in ("Elite", "Strong") and edge >= _CAL_T1_EDGE:
        return "Tier 1"
    if (
        conf in ("High", "Medium")
        and qt in ("Elite", "Strong", "Moderate")
        and edge >= _CAL_T2_EDGE
    ):
        return "Tier 2"
    return "Stay Away"


def calibration_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Compare CLV performance across proxy tier groups.

    Returns ``{tier_label: {"legs": int, "beating_pct": float,
    "avg_clv_prob": float}}``.
    """
    if df.empty:
        return {}

    required = {"confidence_at_pick", "quality_tier_at_pick", "edge_pct_at_pick"}
    if not required.issubset(df.columns):
        return {}

    df = df.copy()
    df["tier_proxy"] = df.apply(_tier_proxy, axis=1)

    result: dict[str, dict] = {}
    for tier in ("Tier 1", "Tier 2", "Stay Away"):
        sub = df[df["tier_proxy"] == tier]
        if sub.empty:
            result[tier] = {"legs": 0, "beating_pct": 0.0, "avg_clv_prob": 0.0}
            continue
        n = len(sub)
        result[tier] = {
            "legs": n,
            "beating_pct": round(100.0 * (sub["clv_prob"] > 0).sum() / n, 1),
            "avg_clv_prob": round(float(sub["clv_prob"].mean()), 4),
        }
    return result

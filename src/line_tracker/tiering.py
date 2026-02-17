"""Pluggable tiering system for bet recommendations.

Provides four methods for assigning bet tiers (Tier 1 / Tier 2 / Tier 3 /
Stay Away) to a slate of BetRecommendation objects:

    - ``"absolute"``   — Legacy thresholds (quality_tier + confidence + edge)
    - ``"percentile"``  — Option A: rank by edge_z percentile within slate
    - ``"hybrid"``      — Option B: max(absolute floor, slate percentile)
    - ``"composite"``   — Option C: EV-weighted composite score

All methods preserve the existing BetRecommendation schema and only set the
``bet_tier`` field.  Core math (EV, Z, volatility, CLV) is never modified.
"""

from __future__ import annotations

from statistics import median as _median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from line_tracker.best_bets import BetRecommendation

# ── Tier labels ───────────────────────────────────────────────────────
TIER_1 = "Tier 1"
TIER_2 = "Tier 2"
TIER_3 = "Tier 3"
STAY_AWAY = "Stay Away"

# ── Absolute-method defaults (legacy mapping) ────────────────────────
_ABS_QUALITY_TIERS_T1 = {"Elite", "Strong", "Moderate"}
_ABS_CONFIDENCE_T1 = {"High", "Medium"}
_ABS_QUALITY_TIERS_T2 = {"Elite", "Strong", "Moderate"}


# =====================================================================
# Helpers
# =====================================================================


def _percentile(values: list[float], pct: float) -> float:
    """Compute *pct*-th percentile (0–100) of a sorted-ascending list.

    Uses linear interpolation (same as numpy ``percentile`` with
    ``method='linear'``).  Returns 0.0 for empty lists.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = (pct / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _min_max_normalize(val: float, lo: float, hi: float) -> float:
    """Normalize *val* to [0, 1] via min-max.  Returns 0 when range is 0."""
    if hi - lo <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


# =====================================================================
# Method: absolute (legacy)
# =====================================================================


def _assign_absolute(
    recs: list[BetRecommendation],
    *,
    edge_floor: float = 0.0,
) -> None:
    """Legacy absolute tiering (mirrors original quality_tier mapping).

    Tier 1: quality_tier in {Elite, Strong, Moderate} AND
             confidence in {High, Medium} AND edge_pct >= edge_floor
    Tier 2: quality_tier in {Elite, Strong, Moderate} AND edge_pct > 0
    Tier 3: edge_pct > 0 AND quality_score >= 40
    Stay Away: everything else
    """
    for rec in recs:
        if (
            rec.quality_tier in _ABS_QUALITY_TIERS_T1
            and rec.confidence in _ABS_CONFIDENCE_T1
            and rec.edge_pct >= edge_floor
        ):
            rec.bet_tier = TIER_1
        elif rec.quality_tier in _ABS_QUALITY_TIERS_T2 and rec.edge_pct > 0:
            rec.bet_tier = TIER_2
        elif rec.edge_pct > 0 and rec.quality_score >= 40:
            rec.bet_tier = TIER_3
        else:
            rec.bet_tier = STAY_AWAY


# =====================================================================
# Method A: percentile (relative strength within slate)
# =====================================================================

# Percentile cutoffs — top 5% → Tier 1, next 15% → Tier 2, next 30% → Tier 3
_PCT_TIER1_CUTOFF = 95  # top 5%
_PCT_TIER2_CUTOFF = 80  # next 15%  (80–95)
_PCT_TIER3_CUTOFF = 50  # next 30%  (50–80)

# Absolute floors — no play gets promoted without meeting these
_PCT_MIN_EDGE_PCT = 0.0  # edge must be positive
_PCT_MIN_QUALITY = 50  # quality_score floor


def _assign_percentile(recs: list[BetRecommendation]) -> None:
    """Option A: pure percentile-based tiering on edge_z within the slate.

    • Tier 1 = top 5% of edge_z (at least 1 play if any pass floors)
    • Tier 2 = next 15%
    • Tier 3 = next 30%
    • Stay Away = bottom 50% or fails absolute floors

    Absolute floors:
        edge_pct > 0, quality_score >= 50
    """
    if not recs:
        return

    z_values = [r.edge_z for r in recs]
    p95 = _percentile(z_values, _PCT_TIER1_CUTOFF)
    p80 = _percentile(z_values, _PCT_TIER2_CUTOFF)
    p50 = _percentile(z_values, _PCT_TIER3_CUTOFF)

    # Track whether at least one Tier 1 was assigned
    any_t1 = False

    for rec in recs:
        passes_floor = (
            rec.edge_pct > _PCT_MIN_EDGE_PCT
            and rec.quality_score >= _PCT_MIN_QUALITY
        )
        if not passes_floor:
            rec.bet_tier = STAY_AWAY
            continue

        if rec.edge_z >= p95:
            rec.bet_tier = TIER_1
            any_t1 = True
        elif rec.edge_z >= p80:
            rec.bet_tier = TIER_2
        elif rec.edge_z >= p50:
            rec.bet_tier = TIER_3
        else:
            rec.bet_tier = STAY_AWAY

    # Guarantee: if no Tier 1 was assigned but at least one rec passes
    # floors, promote the single best-Z rec that passes floors to Tier 1.
    if not any_t1:
        eligible = [r for r in recs if r.edge_pct > _PCT_MIN_EDGE_PCT
                    and r.quality_score >= _PCT_MIN_QUALITY]
        if eligible:
            best = max(eligible, key=lambda r: r.edge_z)
            best.bet_tier = TIER_1


# =====================================================================
# Method B: hybrid (absolute floor + relative percentile)
# =====================================================================

_HYBRID_ABS_Z_FLOOR = 1.5  # absolute Z minimum for Tier 1


def _assign_hybrid(recs: list[BetRecommendation]) -> None:
    """Option B: hybrid of absolute thresholds + slate percentiles.

    Tier 1: Z >= max(absolute_floor, 90th percentile Z on slate)
    Tier 2: Z >= 75th percentile
    Tier 3: Z >= median
    Stay Away: below median or edge_pct <= 0

    This ensures:
    - Strong slates: absolute Z standards dominate
    - Weak slates: relative strength still produces ranked output
    """
    if not recs:
        return

    z_values = [r.edge_z for r in recs]
    p90 = _percentile(z_values, 90)
    p75 = _percentile(z_values, 75)
    p50 = _percentile(z_values, 50)

    t1_cutoff = max(_HYBRID_ABS_Z_FLOOR, p90)

    for rec in recs:
        if rec.edge_pct <= 0:
            rec.bet_tier = STAY_AWAY
        elif rec.edge_z >= t1_cutoff:
            rec.bet_tier = TIER_1
        elif rec.edge_z >= p75:
            rec.bet_tier = TIER_2
        elif rec.edge_z >= p50:
            rec.bet_tier = TIER_3
        else:
            rec.bet_tier = STAY_AWAY


# =====================================================================
# Method C: EV-weighted composite score
# =====================================================================

# Composite weights — justified:
#   edge_z (0.45): most statistically principled metric; incorporates
#       shrinkage, volatility normalization, and effective sample size.
#   edge_pct (0.35): raw probability-point edge — the fundamental
#       betting signal; captures magnitude of opportunity.
#   quality_score (0.20): captures market structure (agreement, coverage,
#       freshness) that edge_z alone may miss.
_CW_EDGE_Z = 0.45
_CW_EDGE_PCT = 0.35
_CW_QUALITY = 0.20

_COMP_TIER1_CUTOFF = 85  # score >= 85th percentile
_COMP_TIER2_CUTOFF = 70
_COMP_TIER3_CUTOFF = 50


def compute_composite_score(
    edge_pct: float,
    edge_z: float,
    quality_score: float,
    *,
    slate_edge_pct_range: tuple[float, float],
    slate_edge_z_range: tuple[float, float],
    slate_quality_range: tuple[float, float],
) -> float:
    """Compute the EV-weighted composite score for a single recommendation.

    All three inputs are min-max normalized within the slate, then
    combined using fixed weights.  Returns a value in [0, 1].
    """
    n_ep = _min_max_normalize(edge_pct, *slate_edge_pct_range)
    n_ez = _min_max_normalize(edge_z, *slate_edge_z_range)
    n_qs = _min_max_normalize(quality_score, *slate_quality_range)
    return _CW_EDGE_Z * n_ez + _CW_EDGE_PCT * n_ep + _CW_QUALITY * n_qs


def _assign_composite(recs: list[BetRecommendation]) -> None:
    """Option C: EV-weighted composite score tiering.

    tier_score = 0.45 * norm(edge_z) + 0.35 * norm(edge_pct) + 0.20 * norm(quality)

    Tier 1: score >= 85th percentile
    Tier 2: 70th–85th
    Tier 3: 50th–70th
    Stay Away: below 50th
    """
    if not recs:
        return

    # Compute slate ranges for normalization
    eps = [r.edge_pct for r in recs]
    ezs = [r.edge_z for r in recs]
    qss = [float(r.quality_score) for r in recs]

    ep_range = (min(eps), max(eps))
    ez_range = (min(ezs), max(ezs))
    qs_range = (min(qss), max(qss))

    # Compute scores
    scores: list[float] = []
    for rec in recs:
        s = compute_composite_score(
            rec.edge_pct,
            rec.edge_z,
            float(rec.quality_score),
            slate_edge_pct_range=ep_range,
            slate_edge_z_range=ez_range,
            slate_quality_range=qs_range,
        )
        scores.append(s)

    # Percentile cutoffs on composite scores
    p85 = _percentile(scores, _COMP_TIER1_CUTOFF)
    p70 = _percentile(scores, _COMP_TIER2_CUTOFF)
    p50 = _percentile(scores, _COMP_TIER3_CUTOFF)

    for rec, score in zip(recs, scores):
        if score >= p85:
            rec.bet_tier = TIER_1
        elif score >= p70:
            rec.bet_tier = TIER_2
        elif score >= p50:
            rec.bet_tier = TIER_3
        else:
            rec.bet_tier = STAY_AWAY


# =====================================================================
# Public API: assign_tiers
# =====================================================================

_METHODS = {
    "absolute": _assign_absolute,
    "percentile": _assign_percentile,
    "hybrid": _assign_hybrid,
    "composite": _assign_composite,
}


def assign_tiers(
    recommendations: list[BetRecommendation],
    method: str = "hybrid",
    *,
    edge_floor: float = 0.0,
) -> list[BetRecommendation]:
    """Assign bet tiers to a slate of recommendations.

    Parameters
    ----------
    recommendations:
        List of ``BetRecommendation`` objects (already populated with
        edge_pct, edge_z, quality_score, quality_tier, confidence).
    method:
        Tiering method — one of ``"absolute"``, ``"percentile"``,
        ``"hybrid"``, or ``"composite"``.
    edge_floor:
        Dynamic edge floor for the ``"absolute"`` method (default 0.0).

    Returns
    -------
    The same list, with ``bet_tier`` set on each recommendation.

    Raises
    ------
    ValueError
        If *method* is not one of the supported methods.
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown tiering method {method!r}. "
            f"Choose from: {sorted(_METHODS)}"
        )

    if method == "absolute":
        _assign_absolute(recommendations, edge_floor=edge_floor)
    else:
        _METHODS[method](recommendations)

    # Propagate bet_tier to attached BestBetResult objects
    for rec in recommendations:
        bbr = getattr(rec, "best_bet_result", None)
        if bbr is not None:
            bbr.bet_tier = rec.bet_tier

    return recommendations


# =====================================================================
# Distribution analysis helpers
# =====================================================================


def slate_distribution_stats(
    recs: list[BetRecommendation],
) -> dict:
    """Compute distribution statistics for a slate of recommendations.

    Returns a dict with summary stats for edge_pct, edge_z, and
    quality_score, including percentiles at 50/60/70/80/85/90/95.
    """
    if not recs:
        return {"n": 0}

    eps = [r.edge_pct for r in recs]
    ezs = [r.edge_z for r in recs]
    qss = [float(r.quality_score) for r in recs]

    pct_levels = [50, 60, 70, 80, 85, 90, 95]

    def _stats(values: list[float], name: str) -> dict:
        n = len(values)
        mn = min(values)
        mx = max(values)
        avg = sum(values) / n
        med = _median(values)
        pcts = {f"p{p}": round(_percentile(values, p), 4) for p in pct_levels}
        return {
            f"{name}_min": round(mn, 4),
            f"{name}_max": round(mx, 4),
            f"{name}_mean": round(avg, 4),
            f"{name}_median": round(med, 4),
            **{f"{name}_{k}": v for k, v in pcts.items()},
        }

    stats: dict = {"n": len(recs)}
    stats.update(_stats(eps, "edge_pct"))
    stats.update(_stats(ezs, "edge_z"))
    stats.update(_stats(qss, "quality_score"))

    # Frequency metrics
    stats["pct_z_ge_1"] = round(
        100.0 * sum(1 for z in ezs if z >= 1.0) / len(ezs), 1
    )
    stats["pct_z_ge_1_5"] = round(
        100.0 * sum(1 for z in ezs if z >= 1.5) / len(ezs), 1
    )
    stats["pct_z_ge_2"] = round(
        100.0 * sum(1 for z in ezs if z >= 2.0) / len(ezs), 1
    )
    stats["pct_edge_positive"] = round(
        100.0 * sum(1 for e in eps if e > 0) / len(eps), 1
    )

    return stats


def tier_frequency_report(
    recs: list[BetRecommendation],
) -> dict:
    """Count plays per tier for an already-tiered slate."""
    counts = {TIER_1: 0, TIER_2: 0, TIER_3: 0, STAY_AWAY: 0}
    edge_sums = {TIER_1: [], TIER_2: [], TIER_3: [], STAY_AWAY: []}
    for rec in recs:
        t = rec.bet_tier or STAY_AWAY
        counts[t] = counts.get(t, 0) + 1
        edge_sums[t] = edge_sums.get(t, [])
        edge_sums[t].append(rec.edge_pct)

    result = {"total": len(recs)}
    for tier in (TIER_1, TIER_2, TIER_3, STAY_AWAY):
        result[f"{tier}_count"] = counts[tier]
        edges = edge_sums[tier]
        result[f"{tier}_mean_edge"] = (
            round(sum(edges) / len(edges), 4) if edges else 0.0
        )
    return result

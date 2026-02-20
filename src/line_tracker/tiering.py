"""Pluggable tiering system for bet recommendations.

Provides five methods for assigning bet tiers (Tier 1 / Tier 2 / Tier 3 /
Stay Away) to a slate of BetRecommendation objects:

    - ``"absolute"``   — Legacy thresholds (quality_tier + confidence + edge)
    - ``"percentile"``  — Option A: rank by edge_z percentile within slate
    - ``"hybrid"``      — Option B: max(absolute floor, slate percentile)
    - ``"composite"``   — Option C: EV-weighted composite score
    - ``"quantile"``    — Self-calibrating quantile/target-volume tiers

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
# Method D: self-calibrating quantile/target-volume tiers
# =====================================================================

# ── Tier-score weights ──────────────────────────────────────────────
# Weighted sum of clipped/normalized signal components.
# Positive signals (higher = better):
#   edge_z        — statistically principled; shrinkage + volatility-adjusted
#   edge_ev_shrunk_pct — EV per $100 after shrinkage (= edge_ev_shrunk * 100)
#   agreement_score  — book consensus dispersion (0–100)
#   books_used       — market depth / coverage
#   quality_score    — composite quality metric (0–100)
# Negative signal (higher = worse):
#   hold_max         — market hold / vig overhead
_QW_EDGE_Z = 0.35
_QW_EDGE_EV_SHRUNK = 0.25
_QW_AGREEMENT = 0.15
_QW_BOOKS_USED = 0.15
_QW_QUALITY = 0.10
_QW_HOLD_PENALTY = 0.15  # subtracted

# Clipping denominators — map raw values into [0, 1] via val / denom
_CLIP_EDGE_Z = 3.0
_CLIP_EDGE_EV_SHRUNK_PCT = 5.0  # EV/$100 after shrinkage
_CLIP_AGREEMENT = 100.0
_CLIP_BOOKS_USED = 8.0
_CLIP_QUALITY = 100.0
_CLIP_HOLD = 0.10  # hold in decimal (10%)

# Longshot guardrail: consensus_prob < 0.20 gets a score penalty
_LONGSHOT_PROB_THRESHOLD = 0.20
_LONGSHOT_PENALTY = 0.10

# Minimum eligibility for quantile pool
_Q_MIN_QUALITY = 40
_Q_MIN_EDGE_SHRUNK_PCT = 0.0  # must be > 0 (positive edge after shrinkage)


def _clip01(val: float, denom: float) -> float:
    """Clip ``val / denom`` to [0, 1]."""
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, val / denom))


def tier_score(
    edge_z: float,
    edge_ev_shrunk_pct: float,
    agreement_score: float,
    books_used: int,
    quality_score: float,
    hold_max: float,
    *,
    consensus_prob: float = 1.0,
    alpha_label: str = "",
) -> float:
    """Compute the quantile tier score for a single candidate.

    Uses a weighted sum of clipped/normalized signal components.
    An optional longshot guardrail penalises low-probability bets
    unless ``alpha_label`` is ``"Strong"``.

    Returns a float (typically in [−0.15, 1.0]).  Higher is better.
    """
    score = (
        _QW_EDGE_Z * _clip01(edge_z, _CLIP_EDGE_Z)
        + _QW_EDGE_EV_SHRUNK * _clip01(edge_ev_shrunk_pct, _CLIP_EDGE_EV_SHRUNK_PCT)
        + _QW_AGREEMENT * _clip01(agreement_score, _CLIP_AGREEMENT)
        + _QW_BOOKS_USED * _clip01(books_used, _CLIP_BOOKS_USED)
        + _QW_QUALITY * _clip01(quality_score, _CLIP_QUALITY)
        - _QW_HOLD_PENALTY * _clip01(hold_max, _CLIP_HOLD)
    )

    # Longshot guardrail: penalise if consensus_prob < threshold
    # unless alpha is Strong (sharp-confirmed value).
    if consensus_prob < _LONGSHOT_PROB_THRESHOLD and alpha_label != "Strong":
        score -= _LONGSHOT_PENALTY

    return round(score, 6)


def _rec_tier_score(rec: BetRecommendation) -> float:
    """Extract fields from a BetRecommendation and compute tier_score."""
    edge_ev_shrunk_pct = getattr(rec, "edge_ev_shrunk", 0.0) * 100.0
    hold = getattr(rec, "market_hold_median", 0.0) / 100.0  # pct → decimal
    consensus_prob = getattr(rec, "consensus_prob", 1.0)
    alpha_label = getattr(rec, "alpha_label", "")
    return tier_score(
        edge_z=rec.edge_z,
        edge_ev_shrunk_pct=edge_ev_shrunk_pct,
        agreement_score=getattr(rec, "agreement_score", 0.0),
        books_used=getattr(rec, "books_used_count", 0),
        quality_score=float(rec.quality_score),
        hold_max=hold,
        consensus_prob=consensus_prob,
        alpha_label=alpha_label,
    )


def assign_tiers_quantile(
    recommendations: list[BetRecommendation],
    *,
    tier1_q: float = 0.10,
    tier2_q: float = 0.35,
    min_candidates: int = 10,
    min_quality: int = _Q_MIN_QUALITY,
) -> dict:
    """Assign tiers via quantile/target-volume bucketing.

    Parameters
    ----------
    recommendations:
        Full candidate pool (all recs for the slate/day).
    tier1_q:
        Top quantile share for Tier 1 (default 0.10 = top 10%).
    tier2_q:
        Cumulative quantile share for Tier 2 (default 0.35 = top 35%).
        Tier 2 spans from the Tier 1 cut to this cumulative share.
    min_candidates:
        Below this count the small-slate fallback policy activates.
    min_quality:
        Minimum ``quality_score`` to enter the eligible pool.

    Returns
    -------
    dict with keys:
        ``tier1_cut``  — score cut point for Tier 1
        ``tier2_cut``  — score cut point for Tier 2
        ``scores``     — list of (index, score) for all recs
        ``n_eligible`` — number of eligible candidates
        ``n_total``    — total recs
        ``fallback``   — True if small-slate fallback was used
    """
    n_total = len(recommendations)
    if not recommendations:
        return {
            "tier1_cut": 0.0,
            "tier2_cut": 0.0,
            "scores": [],
            "n_eligible": 0,
            "n_total": 0,
            "fallback": False,
        }

    # ── Compute scores for all recs ──────────────────────────────────
    scores: list[tuple[int, float]] = []
    for i, rec in enumerate(recommendations):
        scores.append((i, _rec_tier_score(rec)))

    # ── Eligible pool: positive shrunk edge + min quality ────────────
    eligible_indices: set[int] = set()
    for i, rec in enumerate(recommendations):
        shrunk = getattr(rec, "edge_ev_shrunk", 0.0)
        if shrunk > _Q_MIN_EDGE_SHRUNK_PCT and rec.quality_score >= min_quality:
            eligible_indices.add(i)

    eligible_scores = sorted(
        [s for i, s in scores if i in eligible_indices], reverse=True,
    )
    n_eligible = len(eligible_scores)

    # ── Determine cut points ─────────────────────────────────────────
    fallback = False
    if n_eligible == 0:
        # No eligible candidates: everything is Stay Away
        tier1_cut = float("inf")
        tier2_cut = float("inf")
    elif n_eligible < min_candidates:
        # Small-slate fallback: guarantee at least 1 Tier 1 if the top
        # candidate has positive edge after shrinkage.
        fallback = True
        tier1_cut = eligible_scores[0]  # exactly the top score
        # Tier 2 cut: take up to 35% of eligible or at least 1 more
        t2_count = max(1, int(n_eligible * tier2_q))
        t2_idx = min(t2_count, n_eligible - 1)
        tier2_cut = eligible_scores[t2_idx]
    else:
        # Normal quantile bucketing on eligible pool
        t1_idx = max(0, int(n_eligible * tier1_q) - 1)
        t2_idx = max(t1_idx + 1, int(n_eligible * tier2_q) - 1)
        t2_idx = min(t2_idx, n_eligible - 1)
        tier1_cut = eligible_scores[t1_idx]
        tier2_cut = eligible_scores[t2_idx]

    # ── Assign tiers ─────────────────────────────────────────────────
    for i, rec in enumerate(recommendations):
        _, s = scores[i]
        if i not in eligible_indices:
            rec.bet_tier = STAY_AWAY
        elif s >= tier1_cut:
            rec.bet_tier = TIER_1
        elif s >= tier2_cut:
            rec.bet_tier = TIER_2
        else:
            rec.bet_tier = STAY_AWAY

    # Small-slate fallback: verify at least 1 Tier 1 exists when there
    # is at least one positive-edge eligible candidate.
    if fallback and n_eligible > 0:
        t1_count = sum(1 for r in recommendations if r.bet_tier == TIER_1)
        if t1_count == 0:
            # Promote the single best-scoring eligible candidate
            best_idx = max(
                eligible_indices,
                key=lambda idx: scores[idx][1],
            )
            recommendations[best_idx].bet_tier = TIER_1

    return {
        "tier1_cut": round(tier1_cut, 6) if tier1_cut != float("inf") else None,
        "tier2_cut": round(tier2_cut, 6) if tier2_cut != float("inf") else None,
        "scores": scores,
        "n_eligible": n_eligible,
        "n_total": n_total,
        "fallback": fallback,
    }


def _assign_quantile(
    recs: list[BetRecommendation],
    *,
    tier1_q: float = 0.10,
    tier2_q: float = 0.35,
    min_candidates: int = 10,
) -> None:
    """Internal wrapper that calls ``assign_tiers_quantile`` in-place."""
    assign_tiers_quantile(
        recs,
        tier1_q=tier1_q,
        tier2_q=tier2_q,
        min_candidates=min_candidates,
    )


# =====================================================================
# Public API: assign_tiers
# =====================================================================

_METHODS = {
    "absolute": _assign_absolute,
    "percentile": _assign_percentile,
    "hybrid": _assign_hybrid,
    "composite": _assign_composite,
    "quantile": _assign_quantile,
}


def assign_tiers(
    recommendations: list[BetRecommendation],
    method: str = "hybrid",
    *,
    edge_floor: float = 0.0,
    tier1_q: float = 0.10,
    tier2_q: float = 0.35,
    min_candidates: int = 10,
) -> list[BetRecommendation]:
    """Assign bet tiers to a slate of recommendations.

    Parameters
    ----------
    recommendations:
        List of ``BetRecommendation`` objects (already populated with
        edge_pct, edge_z, quality_score, quality_tier, confidence).
    method:
        Tiering method — one of ``"absolute"``, ``"percentile"``,
        ``"hybrid"``, ``"composite"``, or ``"quantile"``.
    edge_floor:
        Dynamic edge floor for the ``"absolute"`` method (default 0.0).
    tier1_q:
        Top quantile share for Tier 1 (``"quantile"`` method only).
    tier2_q:
        Cumulative quantile share for Tier 2 (``"quantile"`` method only).
    min_candidates:
        Small-slate threshold (``"quantile"`` method only).

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
    elif method == "quantile":
        _assign_quantile(
            recommendations,
            tier1_q=tier1_q,
            tier2_q=tier2_q,
            min_candidates=min_candidates,
        )
    else:
        _METHODS[method](recommendations)

    # ── Confidence gating: constrain Tier 1 eligibility ──────────
    # A Tier 1 candidate must satisfy:
    #   consensus_prob >= 0.40  OR  alpha_label == "Strong"
    # If not, demote to Tier 2 (candidate is NOT removed from ranking).
    _CONFIDENCE_GATE_PROB = 0.40
    for rec in recommendations:
        if rec.bet_tier == TIER_1:
            prob = getattr(rec, "consensus_prob", 1.0)
            a_label = getattr(rec, "alpha_label", "")
            if prob < _CONFIDENCE_GATE_PROB and a_label != "Strong":
                rec.bet_tier = TIER_2

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

"""Daily Slate aggregator – surfaces the best bets across all events."""

from __future__ import annotations

import os
import statistics
from collections import Counter
from dataclasses import dataclass, replace

from line_tracker.best_bets import recommend_best_bets
from line_tracker.market_structure import sharp_retail_divergence as _sharp_retail_div
from line_tracker.models import BettingLine, BetType

# ── tier thresholds (EV/$100 space) ───────────────────────────────
# edge_pct is now EV per $100 (= 100 * edge_ev).  All edge thresholds
# below are in that same unit (dollars of EV per $100 wagered).
_TIER1_BASE_EDGE = 2.0   # Tier 1A base edge floor (EV/$100)
_TIER1_SIGMA_MULT = 100.0  # sigma is in prob-space; ×100 → EV/$100
_TIER2_EDGE = 0.0         # Tier 2: any positive EV
_TIER2_EDGE_Z_MIN = 0.0
_MIN_BOOKS = 4
_STALE_THRESHOLD_MIN = 120.0
_EDGE_OUTLIER_THRESHOLD = 8.0  # EV/$100 outlier (was 4.0 in prob-space)
_STAY_AWAY_LIMIT = 15
_STAY_AWAY_HOLD_MAX = 8.0  # hold >= 8% is hard Stay Away

# ── Tier 1A (Institutional / Pro) thresholds ─────────────────────────
_TIER1A_BOOKS_MIN = 6
_TIER1A_HOLD_MAX = 6.0

# ── Tier 1B (Standard / Aggressive) thresholds ──────────────────────
_TIER1B_FLOOR_MIN = 1.0    # minimum 1B edge floor (EV/$100)
_TIER1B_BASE_EDGE = 0.5    # 1B base edge (EV/$100)
_TIER1B_SIGMA_MULT = 100.0 # sigma-in-prob → EV/$100 scale
_TIER1B_BOOKS_MIN = 5
_TIER1B_HOLD_MAX = 7.5

# ── market-quality avoid thresholds ───────────────────────────────────
_AVOID_HOLD_MAX = 7.0  # median book hold% above which market is suspect
_AVOID_NOISE_SIGMA_MIN = 0.05  # volatility sigma for "noisy" flag
_AVOID_NOISE_EDGE_MAX = 2.0  # EV/$100 below which noise matters
_AVOID_DIVERGENCE_MIN = 0.04  # sharp-retail divergence threshold

# ── relaxed Tier 2 display thresholds (used when strict mode OFF) ─────
_RELAXED_TIER2_EDGE = 0.3  # EV/$100
_RELAXED_TIER2_EDGE_Z_MIN = 0.75


# ── parameter pack ─────────────────────────────────────────────────


@dataclass(frozen=True)
class TierThresholds:
    """Configurable thresholds for the tier classification cascade.

    All edge thresholds are in **EV/$100** space (= 100 * edge_ev).
    ``edge_pct`` in entry dicts now equals ``edge_ev_100``.

    Use ``STANDARD_THRESHOLDS`` for default behaviour (broader volume)
    or ``PRO_THRESHOLDS`` for stricter curation.
    """

    mode: str = "Standard"

    # Tier 1A (Institutional)  — all edge values in EV/$100
    tier1a_books_min: int = 6
    tier1a_hold_max: float = 6.0
    tier1a_base_edge: float = 2.0     # $2 EV per $100
    tier1a_sigma_mult: float = 100.0  # prob-sigma × 100 → EV/$100

    # Tier 1B (Standard / Aggressive)
    tier1b_books_min: int = 5
    tier1b_hold_max: float = 7.5
    tier1b_base_edge: float = 0.5     # $0.50 EV per $100
    tier1b_sigma_mult: float = 100.0
    tier1b_floor_min: float = 1.0     # $1 EV per $100 minimum
    # Low-confidence override for Tier 1B
    tier1b_low_conf_edge_z_min: float = 2.0
    tier1b_low_conf_edge_pct_min: float = 2.0  # $2 EV per $100
    # Thin-quality override for Tier 1B
    tier1b_thin_edge_pct_min: float = 3.0      # $3 EV per $100
    tier1b_thin_hold_max: float = 6.5
    tier1b_thin_books_min: int = 6

    # Tier 2
    tier2_edge_min: float = 0.0  # >0 effectively (avoid gate ensures edge>0)
    tier2_edge_z_min: float = 0.0  # 0 = disabled
    tier2_low_conf_edge_z_min: float = 1.5
    tier2_low_conf_edge_pct_min: float = 0.5  # $0.50 EV per $100

    # Hard gates (Stay Away)
    min_books: int = 4
    stay_away_hold_max: float = 8.0
    stale_threshold_min: float = 120.0
    edge_outlier_threshold: float = 8.0  # EV/$100 outlier threshold


STANDARD_THRESHOLDS = TierThresholds()

PRO_THRESHOLDS = TierThresholds(
    mode="Pro",
    # Tier 1A: stricter
    tier1a_base_edge=3.0,
    # Tier 1B: stricter floor, no confidence/quality overrides
    tier1b_base_edge=1.0,
    tier1b_sigma_mult=100.0,
    tier1b_floor_min=1.5,
    tier1b_low_conf_edge_z_min=float("inf"),
    tier1b_low_conf_edge_pct_min=float("inf"),
    tier1b_thin_edge_pct_min=float("inf"),
    # Tier 2: stricter edge floor and edge_z gate, no overrides
    tier2_edge_min=0.5,
    tier2_edge_z_min=1.0,
    tier2_low_conf_edge_z_min=float("inf"),
    tier2_low_conf_edge_pct_min=float("inf"),
)


def thresholds_from_calibration(cal: dict) -> TierThresholds:
    """Map a calibration result dict into a ``TierThresholds`` instance.

    Calibration provides per-tier ``edge_ev_100``, ``edge_z``,
    ``hold_max``, and ``books_min``.  Unspecified fields keep
    Standard defaults.
    """
    t1a = cal.get("tier1a", {})
    t1b = cal.get("tier1b", {})
    t2 = cal.get("tier2", {})

    return TierThresholds(
        mode="Auto",
        # Tier 1A
        tier1a_base_edge=t1a.get("edge_ev_100", 2.0),
        tier1a_hold_max=t1a.get("hold_max", 6.0),
        tier1a_books_min=t1b.get("books_min", 6),
        # Tier 1B
        tier1b_base_edge=t1b.get("edge_ev_100", 0.5),
        tier1b_floor_min=t1b.get("edge_ev_100", 1.0),
        tier1b_hold_max=t1b.get("hold_max", 7.5),
        tier1b_books_min=t1b.get("books_min", 5),
        # Tier 2
        tier2_edge_min=t2.get("edge_ev_100", 0.0),
        tier2_edge_z_min=t2.get("edge_z", 0.0),
        # Hard gates: use t2 hold_max as stay_away_hold_max if wider
        min_books=t2.get("books_min", 4),
    )


def get_thresholds(mode: str = "Standard") -> TierThresholds:
    """Return the threshold pack for the given mode name."""
    if mode == "Pro":
        return PRO_THRESHOLDS
    return STANDARD_THRESHOLDS


# ── dynamic edge floor helpers ──────────────────────────────────────


def dyn_floor_1a(sigma: float) -> float:
    """Dynamic edge floor for Tier 1A in EV/$100.

    ``sigma`` is robust_sigma in probability space.
    Floor = max(base, 100 * sigma) — ensures Tier 1A edge clears
    at least one sigma in EV-space.
    """
    return max(_TIER1_BASE_EDGE, _TIER1_SIGMA_MULT * sigma)


def dyn_floor_1b(sigma: float) -> float:
    """Dynamic edge floor for Tier 1B in EV/$100.

    Floor = max(floor_min, 100 * sigma).
    """
    return max(_TIER1B_FLOOR_MIN, _TIER1B_SIGMA_MULT * sigma)


def compute_distance_to_1b(
    entry: dict,
    thresholds: TierThresholds | None = None,
) -> float:
    """Distance-to-Tier-1B score; lower means closer to qualifying.

    Components:
    * Edge gap below ``dyn_floor_1b``
    * +0.5 if confidence not in (High, Medium)
    * +0.5 if quality_tier not in (Elite, Strong, Moderate)
    * +0.3 if hold > hold_max
    * +0.3 if books < books_min
    """
    th = thresholds or STANDARD_THRESHOLDS
    sigma = entry.get("market_volatility_sigma", 0.0)
    edge = entry.get("edge_pct", 0.0)
    confidence = entry.get("confidence", "")
    quality_tier = entry.get("quality_tier", "")
    hold_median = entry.get("market_hold_median", 0.0)
    books_used = entry.get("books_used", 0)

    floor = max(
        th.tier1b_floor_min,
        th.tier1b_sigma_mult * sigma,
    )

    distance = 0.0
    if edge < floor:
        distance += floor - edge
    if confidence not in ("High", "Medium"):
        distance += 0.5
    if quality_tier not in ("Elite", "Strong", "Moderate"):
        distance += 0.5
    if hold_median > th.tier1b_hold_max:
        distance += 0.3
    if books_used < th.tier1b_books_min:
        distance += 0.3

    return round(distance, 3)


def _slate_score(quality_score: float, edge_pct: float) -> float:
    """0.6 * quality_score + 0.4 * min(edge_pct, 5) * 20"""
    return 0.6 * quality_score + 0.4 * min(edge_pct, 5) * 20


# ── market-quality avoid flags ─────────────────────────────────────


def _add_market_quality_flags(reasons: list[str], entry: dict) -> None:
    """Append market-quality avoid flags to *reasons* (mutates in place)."""
    hold = entry.get("market_hold_median", 0.0)
    sigma = entry.get("market_volatility_sigma", 0.0)
    edge = entry.get("edge_pct", 0.0)
    div = entry.get("divergence")

    if hold > _AVOID_HOLD_MAX:
        reasons.append(
            f"High market hold ({hold:.1f}% > {_AVOID_HOLD_MAX:.0f}%)"
        )
    if sigma > _AVOID_NOISE_SIGMA_MIN and edge < _AVOID_NOISE_EDGE_MAX:
        reasons.append(
            f"Noisy market (\u03c3={sigma:.4f}, edge only {edge:.1f}%)"
        )
    if div is not None and div > _AVOID_DIVERGENCE_MIN:
        reasons.append(
            f"Sharp-retail divergence ({div:.4f} > {_AVOID_DIVERGENCE_MIN})"
        )


def _avoid_score(entry: dict) -> float:
    """Composite badness score — higher means more reasons to avoid."""
    score = 0.0

    conf = entry.get("confidence", "")
    if conf == "Low":
        score += 30.0
    elif conf not in ("High", "Medium"):
        score += 20.0

    qt = entry.get("quality_tier", "")
    if qt == "Thin":
        score += 25.0
    elif qt not in ("Elite", "Strong", "Moderate"):
        score += 15.0

    edge_z = entry.get("edge_z", 0.0)
    if edge_z and edge_z < 1.0:
        score += max(0.0, 20.0 * (1.0 - edge_z))

    hold = entry.get("market_hold_median", 0.0)
    if hold > _AVOID_HOLD_MAX:
        score += 15.0

    sigma = entry.get("market_volatility_sigma", 0.0)
    edge = entry.get("edge_pct", 0.0)
    if sigma > _AVOID_NOISE_SIGMA_MIN and edge < _AVOID_NOISE_EDGE_MAX:
        score += 20.0

    div = entry.get("divergence")
    if div is not None and div > _AVOID_DIVERGENCE_MIN:
        score += 15.0

    return round(score, 2)


# ── classification ─────────────────────────────────────────────────


def classify_rec(
    entry: dict,
    settings: dict | None = None,
    thresholds: TierThresholds | None = None,
) -> dict:
    """Classify a recommendation into tier1a/tier1b/tier2/tier3/avoid.

    Runs for **every** recommendation — nothing is skipped before
    classification.

    Tier cascade:
        1. Hard disqualifiers → **avoid** (unstable, <4 books, stale, outlier)
        2. hold >= 8% or edge <= 0 → **avoid**
        3. Tier 1A (Institutional): High conf, Elite/Strong quality,
           books >= 6, hold <= 6%, edge >= dyn_floor_1a
        4. Tier 1B (Standard): High/Medium conf (or Low with override),
           Elite/Strong/Moderate quality (or Thin with override),
           books >= 5, hold <= 7.5%, edge >= dyn_floor_1b
        5. Tier 2: High/Medium conf (or Low with override),
           Elite/Strong/Moderate quality, edge > 0, books >= 4
        6. Tier 3: positive-edge plays that don't meet higher tiers

    Parameters
    ----------
    entry:
        A dict with at least: edge_pct, confidence, quality_tier,
        quality_score, market_volatility_sigma, edge_z, books_used,
        oldest_update_age_min, market_unstable, market_hold_median.
    settings:
        Reserved for future use.
    thresholds:
        A ``TierThresholds`` parameter pack.  Defaults to
        ``STANDARD_THRESHOLDS``.

    Returns
    -------
    dict with keys:
        tier: ``"tier1a"`` | ``"tier1b"`` | ``"tier2"`` | ``"tier3"``
              | ``"avoid"``
        reasons: list[str]  (empty except for avoid)
        dynamic_edge_floor: float  (Tier 1A floor, for debugging)
    """
    settings = settings or {}
    th = thresholds or STANDARD_THRESHOLDS

    edge = entry.get("edge_pct", 0.0)
    confidence = entry.get("confidence", "")
    quality_tier = entry.get("quality_tier", "")
    sigma = entry.get("market_volatility_sigma", 0.0)
    edge_z = entry.get("edge_z", 0.0)
    market_unstable = entry.get("market_unstable", False)
    books_used = entry.get("books_used", 0)
    oldest_age = entry.get("oldest_update_age_min", 0.0)
    hold_median = entry.get("market_hold_median", 0.0)

    floor_1a = max(th.tier1a_base_edge, th.tier1a_sigma_mult * sigma)

    # ── Hard disqualifiers (block ALL tiers → Stay Away) ──────────
    hard_reasons: list[str] = []
    if market_unstable:
        hard_reasons.append("Unstable market (too many outlier books filtered)")
    if books_used and books_used < th.min_books:
        hard_reasons.append(f"Too few books (<{th.min_books})")
    if oldest_age > th.stale_threshold_min:
        hard_reasons.append(
            f"Stale lines (oldest update > {th.stale_threshold_min:.0f} min)"
        )
    if edge >= th.edge_outlier_threshold and confidence == "Low":
        hard_reasons.append("Edge outlier with low confidence (possible bad data)")

    if hard_reasons:
        _add_market_quality_flags(hard_reasons, entry)
        return {
            "tier": "avoid",
            "reasons": hard_reasons,
            "dynamic_edge_floor": floor_1a,
        }

    # ── Hold gate → Stay Away ─────────────────────────────────────
    if hold_median >= th.stay_away_hold_max:
        reasons = [
            f"Market hold too high "
            f"({hold_median:.1f}% >= {th.stay_away_hold_max:.0f}%)"
        ]
        _add_market_quality_flags(reasons, entry)
        return {"tier": "avoid", "reasons": reasons, "dynamic_edge_floor": floor_1a}

    # ── Negative / zero edge → Stay Away ──────────────────────────
    if edge <= 0:
        reasons = [f"Edge not positive ({edge:.1f}%)"]
        _add_market_quality_flags(reasons, entry)
        return {"tier": "avoid", "reasons": reasons, "dynamic_edge_floor": floor_1a}

    # ── Tier 1A (Institutional / Pro) ─────────────────────────────
    if (
        confidence == "High"
        and quality_tier in ("Elite", "Strong")
        and edge >= floor_1a
        and books_used >= th.tier1a_books_min
        and hold_median <= th.tier1a_hold_max
    ):
        return {"tier": "tier1a", "reasons": [], "dynamic_edge_floor": floor_1a}

    # ── Tier 1B (Standard / Aggressive) ───────────────────────────
    # Confidence: High/Medium by default, Low allowed with override
    t1b_conf = confidence in ("High", "Medium")
    if not t1b_conf and confidence == "Low":
        if (
            edge_z >= th.tier1b_low_conf_edge_z_min
            and edge >= th.tier1b_low_conf_edge_pct_min
        ):
            t1b_conf = True

    # Quality: Elite/Strong/Moderate by default, Thin allowed with override
    t1b_qt = quality_tier in ("Elite", "Strong", "Moderate")
    if not t1b_qt and quality_tier == "Thin":
        if (
            edge >= th.tier1b_thin_edge_pct_min
            and hold_median <= th.tier1b_thin_hold_max
            and books_used >= th.tier1b_thin_books_min
        ):
            t1b_qt = True

    floor_1b = max(
        th.tier1b_floor_min,
        th.tier1b_sigma_mult * sigma,
    )
    if (
        t1b_conf
        and t1b_qt
        and edge >= floor_1b
        and books_used >= th.tier1b_books_min
        and hold_median <= th.tier1b_hold_max
    ):
        return {"tier": "tier1b", "reasons": [], "dynamic_edge_floor": floor_1a}

    # ── Tier 2 ────────────────────────────────────────────────────
    # Confidence: High/Medium by default, Low allowed with override
    t2_conf = confidence in ("High", "Medium")
    if not t2_conf and confidence == "Low":
        if (
            edge_z >= th.tier2_low_conf_edge_z_min
            and edge >= th.tier2_low_conf_edge_pct_min
        ):
            t2_conf = True

    t2_qt = quality_tier in ("Elite", "Strong", "Moderate")
    t2_edge = edge >= th.tier2_edge_min if th.tier2_edge_min > 0 else True
    # edge_z gate: disabled when tier2_edge_z_min == 0
    if th.tier2_edge_z_min > 0:
        t2_ez = edge_z >= th.tier2_edge_z_min if edge_z else True
    else:
        t2_ez = True
    t2_books = books_used >= th.min_books

    if t2_conf and t2_qt and t2_edge and t2_ez and t2_books:
        return {"tier": "tier2", "reasons": [], "dynamic_edge_floor": floor_1a}

    # ── Tier 3: positive edge, no hard Stay Away flags ────────────
    # Edge > 0 already guaranteed (checked above).
    return {"tier": "tier3", "reasons": [], "dynamic_edge_floor": floor_1a}


# ── Stay Away ranking ──────────────────────────────────────────────


def _stay_away_sort_key(entry: dict) -> tuple:
    """Composite key: worst entries sort first (ascending).

    Primary: highest avoid_score first (negated so ascending = worst).
    Secondary tiebreakers: confidence, edge_z, sigma, quality_score.
    """
    conf_order = {"Low": 0, "Medium": 1, "High": 2}
    return (
        -_avoid_score(entry),
        conf_order.get(entry.get("confidence", ""), 1),
        entry.get("edge_z", 0.0),
        -entry.get("market_volatility_sigma", 0.0),
        entry.get("quality_score", 0),
    )


# ── relaxed Tier 2 display check (post-classification) ───────────────


def passes_relaxed_tier2(entry: dict) -> bool:
    """Return True if a stay-away entry meets relaxed Tier 2 display criteria.

    Called by the dashboard when *strict mode* is OFF.  This never changes
    the canonical tier assignment — it only decides whether to **show** the
    entry in the Tier 2 section instead of Stay Away.

    Relaxed rules (vs. strict):
    * edge >= 0.5 %  (strict: 1.5 %)
    * edge_z >= 0.75  (strict: 1.0), skipped when 0
    * confidence Low allowed **only** when quality_tier in (Elite, Strong)
    * Hard disqualifiers still block promotion.
    """
    # Hard disqualifiers — never promote
    if entry.get("market_unstable", False):
        return False
    books = entry.get("books_used", 0)
    if books and books < _MIN_BOOKS:
        return False
    if entry.get("oldest_update_age_min", 0.0) > _STALE_THRESHOLD_MIN:
        return False
    edge = entry.get("edge_pct", 0.0)
    if edge >= _EDGE_OUTLIER_THRESHOLD and entry.get("confidence", "") == "Low":
        return False

    # Relaxed thresholds
    if edge <= 0 or edge < _RELAXED_TIER2_EDGE:
        return False
    edge_z = entry.get("edge_z", 0.0)
    if edge_z and edge_z < _RELAXED_TIER2_EDGE_Z_MIN:
        return False
    confidence = entry.get("confidence", "")
    quality_tier = entry.get("quality_tier", "")
    if confidence == "Low" and quality_tier not in ("Elite", "Strong"):
        return False
    if confidence not in ("High", "Medium", "Low"):
        return False
    if quality_tier not in ("Elite", "Strong", "Moderate"):
        return False
    return True


# ── debug statistics ──────────────────────────────────────────────────


def _min_med_max(vals: list[float]) -> dict:
    """Return min / median / max for a list of floats (empty-safe)."""
    if not vals:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }


def compute_slate_debug_stats(entries: list[dict]) -> dict:
    """Compute detailed debug statistics over *all* classified entries.

    Parameters
    ----------
    entries:
        The flat list of entry dicts **after** ``classify_rec`` has run
        (each entry has ``tier``, ``dynamic_edge_floor``, etc.).

    Returns
    -------
    dict with keys:
        total_recs, counts_by_tier, counts_by_confidence,
        counts_by_quality_tier, tier1_gate_failures, tier2_gate_failures,
        sigma_stats, dynamic_floor_stats, warnings.
    """
    total = len(entries)

    counts_by_tier = dict(Counter(e["tier"] for e in entries))
    counts_by_confidence = dict(Counter(e.get("confidence", "") for e in entries))
    counts_by_quality_tier = dict(Counter(e.get("quality_tier", "") for e in entries))

    # ── Tier 1 gate failure counts ────────────────────────────────
    t1_fail_conf = 0
    t1_fail_qt = 0
    t1_fail_dyn_floor = 0
    t1_fail_edge_pos = 0

    # ── Tier 2 gate failure counts ────────────────────────────────
    t2_fail_conf = 0
    t2_fail_qt = 0
    t2_fail_edge_floor = 0
    t2_fail_edge_z = 0

    sigmas: list[float] = []
    robust_sigmas: list[float] = []
    dyn_floors: list[float] = []

    for e in entries:
        edge = e.get("edge_pct", 0.0)
        conf = e.get("confidence", "")
        qt = e.get("quality_tier", "")
        sigma = e.get("market_volatility_sigma", 0.0)
        edge_z = e.get("edge_z", 0.0)
        dfloor = e.get("dynamic_edge_floor", _TIER1_BASE_EDGE)

        sigmas.append(sigma)
        if e.get("robust_sigma") is not None:
            robust_sigmas.append(e["robust_sigma"])
        dyn_floors.append(dfloor)

        # Tier 1 gate failures (count how many entries fail each gate)
        if conf != "High":
            t1_fail_conf += 1
        if qt not in ("Elite", "Strong"):
            t1_fail_qt += 1
        if edge < dfloor:
            t1_fail_dyn_floor += 1
        if edge <= 0:
            t1_fail_edge_pos += 1

        # Tier 2 gate failures
        if conf not in ("High", "Medium"):
            t2_fail_conf += 1
        if qt not in ("Elite", "Strong", "Moderate"):
            t2_fail_qt += 1
        if edge < _TIER2_EDGE:
            t2_fail_edge_floor += 1
        if edge_z and edge_z < _TIER2_EDGE_Z_MIN:
            t2_fail_edge_z += 1

    sigma_stats = _min_med_max(sigmas)
    robust_sigma_stats = _min_med_max(robust_sigmas)
    floor_stats = _min_med_max(dyn_floors)

    # ── Sanity warnings ───────────────────────────────────────────
    warnings: list[str] = []
    n_tier1a = counts_by_tier.get("tier1a", 0)
    n_tier1b = counts_by_tier.get("tier1b", 0)
    n_tier2 = counts_by_tier.get("tier2", 0)
    n_tier3 = counts_by_tier.get("tier3", 0)
    n_avoid = counts_by_tier.get("avoid", 0)
    n_all = n_tier1a + n_tier1b + n_tier2 + n_tier3 + n_avoid

    if total > 0 and n_all == 0:
        warnings.append(
            "total_recs > 0 but all tier counts == 0 "
            "(entries may be lost)"
        )
    if total > 0 and n_avoid == 0 and n_tier3 == 0:
        warnings.append(
            "total_recs > 0 but StayAway+Tier3 == 0 "
            "(every rec passed — check thresholds)"
        )
    if sigma_stats["max"] is not None and sigma_stats["max"] > 0.25:
        warnings.append(
            f"sigma max = {sigma_stats['max']:.4f} > 0.25 "
            "(likely units bug — sigma should be in probability units)"
        )
    if robust_sigma_stats["max"] is not None and robust_sigma_stats["max"] > 0.25:
        warnings.append(
            f"robust_sigma max = {robust_sigma_stats['max']:.4f} > 0.25 "
            "(likely units bug)"
        )
    if floor_stats["median"] is not None and floor_stats["median"] > 6.0:
        warnings.append(
            f"dynamic floor median = {floor_stats['median']:.2f}% > 6.0% "
            "(likely units bug — sigma may be in wrong units)"
        )

    return {
        "total_recs": total,
        "counts_by_tier": counts_by_tier,
        "counts_by_confidence": counts_by_confidence,
        "counts_by_quality_tier": counts_by_quality_tier,
        "tier1_gate_failures": {
            "confidence_not_high": t1_fail_conf,
            "quality_tier_not_elite_strong": t1_fail_qt,
            "below_dynamic_floor": t1_fail_dyn_floor,
            "edge_not_positive": t1_fail_edge_pos,
        },
        "tier2_gate_failures": {
            "confidence_not_high_medium": t2_fail_conf,
            "quality_tier_not_elite_strong_moderate": t2_fail_qt,
            "below_edge_floor": t2_fail_edge_floor,
            "edge_z_too_low": t2_fail_edge_z,
        },
        "sigma_stats": sigma_stats,
        "robust_sigma_stats": robust_sigma_stats,
        "dynamic_floor_stats": floor_stats,
        "warnings": warnings,
    }


# ── display filters (applied AFTER classification) ─────────────────


def _passes_filters(entry: dict, filters: dict) -> bool:
    """Return True if the entry survives all user-supplied display filters."""
    if filters.get("min_edge") and entry["edge_pct"] < filters["min_edge"]:
        return False
    if filters.get("min_quality") and entry["quality_score"] < filters["min_quality"]:
        return False
    if filters.get("markets") and entry["market"] not in filters["markets"]:
        return False
    if filters.get("hide_low_confidence") and entry["confidence"] == "Low":
        return False
    if (
        filters.get("books_used_min")
        and entry["books_used"] < filters["books_used_min"]
    ):
        return False
    return True


# ── slate builder ──────────────────────────────────────────────────


def build_daily_slate(
    lines_by_event: dict[str, list[BettingLine]],
    *,
    filters: dict | None = None,
    settings: dict | None = None,
    thresholds: TierThresholds | None = None,
) -> dict:
    """Build the daily slate from lines grouped by event.

    Every recommendation is classified first
    (tier1a / tier1b / tier2 / tier3 / avoid).  User display-filters
    are applied only to actionable tier lists afterwards — Stay Away
    is always populated when recs exist.

    Returns
    -------
    dict with keys:
        ``"tier1a"``, ``"tier1b"``, ``"tier1"`` (1a+1b combined),
        ``"tier2"``, ``"tier3"``, ``"stay_away"``,
        ``"closest_candidates"`` (Tier 2/3 entries closest to Tier 1B),
        ``"counts"``, ``"debug_stats"``, and optional ``"debug"``.
    """
    filters = filters or {}
    settings = settings or {}
    th = thresholds or STANDARD_THRESHOLDS
    max_per_event: int = filters.get("max_per_event", 1)

    all_entries: list[dict] = []

    for event_id, lines in lines_by_event.items():
        if not lines:
            continue

        recs = recommend_best_bets(lines)
        if not recs:
            continue

        top_recs = recs[: max(1, max_per_event)]

        # Pre-compute sharp-retail divergence per market for this event
        _div_cache: dict[str, float | None] = {}
        for bt in BetType:
            mkt_lines = [ln for ln in lines if ln.bet_type == bt]
            if len(mkt_lines) >= 2:
                div_info = _sharp_retail_div(mkt_lines)
                _div_cache[bt.value] = div_info.get("divergence")

        sample_line = lines[0]
        event_name = sample_line.event
        commence_time = sample_line.commence_time

        for rec in top_recs:
            score = _slate_score(rec.quality_score, rec.edge_pct)

            entry: dict = {
                "event_id": event_id,
                "event": event_name,
                "commence_time": commence_time,
                "market": rec.market,
                "selection": rec.selection,
                "line": rec.line,
                "best_odds": rec.best_odds,
                "best_sportsbook": rec.best_sportsbook,
                "edge_pct": rec.ev_100,  # EV/$100 — tiering key
                "ev_100": rec.ev_100,
                "ev_roi": rec.ev_roi,
                "edge_pp": rec.edge_pp,
                "edge_pct_pp": rec.edge_pct,  # prob-point %
                "edge_ev": rec.edge_ev,
                "edge_ev_shrunk": rec.edge_ev_shrunk,
                "edge_ev_100": rec.edge_ev_100,
                "n_eff": rec.n_eff,
                "outlier_rate": rec.outlier_rate,
                "ev_sigma": rec.ev_sigma,
                "consensus_prob_weighted": rec.consensus_prob_weighted,
                "confidence": rec.confidence,
                "quality_score": rec.quality_score,
                "quality_tier": rec.quality_tier,
                "market_volatility_sigma": rec.market_volatility_sigma,
                "robust_sigma": rec.robust_sigma,
                "edge_z": rec.edge_z,
                "market_unstable": rec.market_unstable,
                "slate_score": score,
                "books_used": rec.books_used_count,
                "updated_age_min": rec.newest_update_age_min,
                "oldest_update_age_min": rec.oldest_update_age_min,
                "market_hold_median": rec.market_hold_median,
                "divergence": _div_cache.get(rec.market),
                "kelly_base": rec.kelly_base,
                "kelly_suggested": rec.kelly_suggested,
                "sizing_note": rec.sizing_note,
            }

            # Classify — runs for EVERY rec, no pre-filtering
            classification = classify_rec(
                entry, settings=settings, thresholds=th,
            )
            entry["tier"] = classification["tier"]
            entry["avoid_reasons"] = classification["reasons"]
            entry["dynamic_edge_floor"] = classification["dynamic_edge_floor"]
            entry["avoid_score"] = _avoid_score(entry)
            entry["distance_to_1b"] = compute_distance_to_1b(
                entry, thresholds=th,
            )

            all_entries.append(entry)

    # Sort by slate_score descending
    all_entries.sort(key=lambda e: e["slate_score"], reverse=True)

    # ── Bucket into tiers ─────────────────────────────────────────
    _top_tiers = {"tier1a", "tier1b"}
    result: dict = {
        "tier1a": [], "tier1b": [], "tier1": [],
        "tier2": [], "tier3": [], "stay_away": [],
        "closest_candidates": [],
    }
    display_filters = {k: v for k, v in filters.items() if k != "max_per_event"}

    for entry in all_entries:
        tier = entry["tier"]
        if tier in _top_tiers:
            if _passes_filters(entry, display_filters):
                result[tier].append(entry)
                result["tier1"].append(entry)
        elif tier == "tier2":
            if _passes_filters(entry, display_filters):
                result["tier2"].append(entry)
        elif tier == "tier3":
            if _passes_filters(entry, display_filters):
                result["tier3"].append(entry)
        else:
            # Stay Away: only apply markets filter
            markets_filter = {}
            if "markets" in filters:
                markets_filter["markets"] = filters["markets"]
            if _passes_filters(entry, markets_filter):
                result["stay_away"].append(entry)

    # Rank Stay Away by "worst-ness" and cap at _STAY_AWAY_LIMIT
    result["stay_away"].sort(key=_stay_away_sort_key)
    result["stay_away"] = result["stay_away"][:_STAY_AWAY_LIMIT]

    # ── Closest candidates (tier2/tier3 entries nearest to Tier 1B) ─
    candidate_pool = result["tier2"] + result["tier3"]
    for e in candidate_pool:
        e.setdefault("distance_to_1b", compute_distance_to_1b(e, thresholds=th))
    result["closest_candidates"] = sorted(
        candidate_pool, key=lambda e: e["distance_to_1b"],
    )[:10]

    # ── Counts (always available) ─────────────────────────────────
    result["counts"] = {
        "total_recs": len(all_entries),
        "tier1a": sum(1 for e in all_entries if e["tier"] == "tier1a"),
        "tier1b": sum(1 for e in all_entries if e["tier"] == "tier1b"),
        "tier1": sum(
            1 for e in all_entries if e["tier"] in ("tier1a", "tier1b")
        ),
        "tier2": sum(1 for e in all_entries if e["tier"] == "tier2"),
        "tier3": sum(1 for e in all_entries if e["tier"] == "tier3"),
        "stay_away": sum(1 for e in all_entries if e["tier"] == "avoid"),
    }

    # ── Debug stats (always computed, keyed separately) ──────────
    result["debug_stats"] = compute_slate_debug_stats(all_entries)

    # ── Debug counters (extended breakdown, shown only on flag) ───
    show_debug = (
        os.environ.get("LINE_TRACKER_DEBUG", "").lower() in ("1", "true", "yes")
        or filters.get("debug", False)
    )
    if show_debug:
        result["debug"] = {
            **result["counts"],
            "total_recs": len(all_entries),
            "tier1a_count": result["counts"]["tier1a"],
            "tier1b_count": result["counts"]["tier1b"],
            "tier1_count": result["counts"]["tier1"],
            "tier2_count": result["counts"]["tier2"],
            "tier3_count": result["counts"]["tier3"],
            "stay_away_count": result["counts"]["stay_away"],
            "by_confidence": dict(
                Counter(e.get("confidence", "") for e in all_entries)
            ),
            "by_quality_tier": dict(
                Counter(e.get("quality_tier", "") for e in all_entries)
            ),
        }
        result["volume_tuning"] = compute_volume_tuning_stats(all_entries)

    # ── Thresholds used (always available) ─────────────────────────
    result["thresholds"] = th

    return result


# ── volume tuning ─────────────────────────────────────────────────


def _percentiles(
    vals: list[float],
    pcts: tuple[int, ...] = (10, 50, 90),
) -> dict:
    """Return {pN: value} for given percentiles (empty-safe)."""
    if not vals:
        return {f"p{p}": None for p in pcts}
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    result = {}
    for p in pcts:
        idx = max(0, min(n - 1, int(p / 100 * n)))
        result[f"p{p}"] = sorted_vals[idx]
    return result


def compute_volume_tuning_stats(entries: list[dict]) -> dict:
    """Compute detailed volume tuning statistics.

    Returns distribution snapshots and percentiles (p10/p50/p90)
    for key metrics, plus histograms for categorical fields.
    """
    if not entries:
        return {"total": 0}

    edges = [e.get("edge_pct", 0.0) for e in entries]
    edge_zs = [e.get("edge_z", 0.0) for e in entries if e.get("edge_z", 0.0)]
    sigmas = [e.get("market_volatility_sigma", 0.0) for e in entries]
    holds = [e.get("market_hold_median", 0.0) for e in entries]
    books = [e.get("books_used", 0) for e in entries]

    # Categorical distributions
    conf_dist = dict(Counter(e.get("confidence", "") for e in entries))
    qt_dist = dict(Counter(e.get("quality_tier", "") for e in entries))
    tier_dist = dict(Counter(e.get("tier", "") for e in entries))

    # Books histogram (buckets: 1-3, 4-5, 6-7, 8+)
    books_hist = {"1-3": 0, "4-5": 0, "6-7": 0, "8+": 0}
    for b in books:
        if b <= 3:
            books_hist["1-3"] += 1
        elif b <= 5:
            books_hist["4-5"] += 1
        elif b <= 7:
            books_hist["6-7"] += 1
        else:
            books_hist["8+"] += 1

    return {
        "total": len(entries),
        "counts_by_tier": tier_dist,
        "confidence_dist": conf_dist,
        "quality_tier_dist": qt_dist,
        "books_histogram": books_hist,
        "edge_pct": _percentiles(edges),
        "edge_z": _percentiles(edge_zs),
        "sigma": _percentiles(sigmas),
        "hold_median": _percentiles(holds),
        "books_used": _percentiles([float(b) for b in books]),
    }


# ── threshold auto-tuning ─────────────────────────────────────────


def suggest_thresholds(
    summary_stats: dict,
    base: TierThresholds | None = None,
) -> TierThresholds:
    """Suggest relaxed thresholds when Tier 1A is underpopulated.

    Applies ordered relaxations when Tier 1A is empty:
        1. ``books_used`` min **-1** (floor 5)
        2. ``hold`` max **+0.5** (cap 7.0)
        3. dyn floor intercept **-0.25** (floor 2.0)
        4. sigma mult **-0.1** (floor 0.7)

    Parameters
    ----------
    summary_stats:
        Output of ``compute_slate_debug_stats`` (needs ``counts_by_tier``).
    base:
        Starting thresholds to relax from (default ``STANDARD_THRESHOLDS``).

    Returns
    -------
    A new ``TierThresholds`` with relaxed Tier 1A values, or the
    original if 1A is already populated.
    """
    th = base or STANDARD_THRESHOLDS
    counts = summary_stats.get("counts_by_tier", {})
    n_tier1a = counts.get("tier1a", 0)

    if n_tier1a > 0:
        return th  # no relaxation needed

    return replace(
        th,
        tier1a_books_min=max(5, th.tier1a_books_min - 1),
        tier1a_hold_max=min(7.0, th.tier1a_hold_max + 0.5),
        tier1a_base_edge=max(1.5, th.tier1a_base_edge - 0.25),
        tier1a_sigma_mult=max(50.0, th.tier1a_sigma_mult - 10.0),
    )


# ── CLI summary ───────────────────────────────────────────────────


def print_slate_summary(slate: dict) -> str:
    """Format a human-readable slate summary for CLI output."""
    counts = slate.get("counts", {})
    th = slate.get("thresholds", STANDARD_THRESHOLDS)
    lines = [
        f"Mode: {th.mode}",
        f"Total recs: {counts.get('total_recs', 0)}",
        f"  Tier 1A: {counts.get('tier1a', 0)}",
        f"  Tier 1B: {counts.get('tier1b', 0)}",
        f"  Tier 1 (combined): {counts.get('tier1', 0)}",
        f"  Tier 2: {counts.get('tier2', 0)}",
        f"  Tier 3: {counts.get('tier3', 0)}",
        f"  Stay Away: {counts.get('stay_away', 0)}",
    ]
    vt = slate.get("volume_tuning")
    if vt and vt.get("total", 0) > 0:
        lines.append("")
        lines.append("Volume Tuning:")
        for key in ("edge_pct", "edge_z", "sigma", "hold_median", "books_used"):
            pcts = vt.get(key, {})
            lines.append(
                f"  {key}: "
                f"p10={pcts.get('p10')}, p50={pcts.get('p50')}, p90={pcts.get('p90')}"
            )
        bh = vt.get("books_histogram", {})
        lines.append(f"  books_histogram: {bh}")
    return "\n".join(lines)

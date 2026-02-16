"""Daily Slate aggregator – surfaces the best bets across all events."""

from __future__ import annotations

import os
import statistics
from collections import Counter

from line_tracker.best_bets import recommend_best_bets
from line_tracker.market_structure import sharp_retail_divergence as _sharp_retail_div
from line_tracker.models import BettingLine, BetType

# ── tier thresholds ────────────────────────────────────────────────
_TIER1_BASE_EDGE = 2.5   # Tier 1A base edge floor (relaxed from 3.0)
_TIER1_SIGMA_MULT = 1.0  # Tier 1A sigma multiplier (relaxed from 1.2)
_TIER2_EDGE = 1.0         # Tier 2 static edge floor (relaxed from 1.5)
_TIER2_EDGE_Z_MIN = 1.0
_MIN_BOOKS = 4
_STALE_THRESHOLD_MIN = 120.0
_EDGE_OUTLIER_THRESHOLD = 4.0
_STAY_AWAY_LIMIT = 15
_STAY_AWAY_HOLD_MAX = 8.0  # hold >= 8% is hard Stay Away

# ── Tier 1A (Institutional / Pro) thresholds ─────────────────────────
_TIER1A_BOOKS_MIN = 6
_TIER1A_HOLD_MAX = 6.0

# ── Tier 1B (Standard / Aggressive) thresholds ──────────────────────
_TIER1B_FLOOR_MIN = 2.0   # minimum 1B edge floor
_TIER1B_BASE_EDGE = 1.0
_TIER1B_SIGMA_MULT = 1.0
_TIER1B_BOOKS_MIN = 5
_TIER1B_HOLD_MAX = 7.5

# ── market-quality avoid thresholds ───────────────────────────────────
_AVOID_HOLD_MAX = 7.0  # median book hold% above which market is suspect
_AVOID_NOISE_SIGMA_MIN = 0.05  # volatility sigma for "noisy" flag
_AVOID_NOISE_EDGE_MAX = 2.0  # edge% below which noise matters
_AVOID_DIVERGENCE_MIN = 0.04  # sharp-retail divergence threshold

# ── relaxed Tier 2 display thresholds (used when strict mode OFF) ─────
_RELAXED_TIER2_EDGE = 0.5
_RELAXED_TIER2_EDGE_Z_MIN = 0.75


# ── dynamic edge floor helpers ──────────────────────────────────────


def dyn_floor_1a(sigma: float) -> float:
    """Dynamic edge floor for Tier 1A: base + mult * sigma."""
    return _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma


def dyn_floor_1b(sigma: float) -> float:
    """Dynamic edge floor for Tier 1B: max(floor_min, base + mult * sigma)."""
    return max(_TIER1B_FLOOR_MIN, _TIER1B_BASE_EDGE + _TIER1B_SIGMA_MULT * sigma)


def compute_distance_to_1b(entry: dict) -> float:
    """Distance-to-Tier-1B score; lower means closer to qualifying.

    Components:
    * Edge gap below ``dyn_floor_1b``
    * +0.5 if confidence not in (High, Medium)
    * +0.5 if quality_tier not in (Elite, Strong, Moderate)
    * +0.3 if hold > ``_TIER1B_HOLD_MAX``
    * +0.3 if books < ``_TIER1B_BOOKS_MIN``
    """
    sigma = entry.get("market_volatility_sigma", 0.0)
    edge = entry.get("edge_pct", 0.0)
    confidence = entry.get("confidence", "")
    quality_tier = entry.get("quality_tier", "")
    hold_median = entry.get("market_hold_median", 0.0)
    books_used = entry.get("books_used", 0)

    floor = dyn_floor_1b(sigma)

    distance = 0.0
    if edge < floor:
        distance += floor - edge
    if confidence not in ("High", "Medium"):
        distance += 0.5
    if quality_tier not in ("Elite", "Strong", "Moderate"):
        distance += 0.5
    if hold_median > _TIER1B_HOLD_MAX:
        distance += 0.3
    if books_used < _TIER1B_BOOKS_MIN:
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


def classify_rec(entry: dict, settings: dict | None = None) -> dict:
    """Classify a recommendation into tier1a/tier1b/tier2/tier3/avoid.

    Runs for **every** recommendation — nothing is skipped before
    classification.

    Tier cascade:
        1. Hard disqualifiers → **avoid** (unstable, <4 books, stale, outlier)
        2. hold >= 8% or edge <= 0 → **avoid**
        3. Tier 1A (Institutional): High conf, Elite/Strong quality,
           books >= 6, hold <= 6%, edge >= dyn_floor_1a
        4. Tier 1B (Standard): High/Medium conf, Elite/Strong/Moderate quality,
           books >= 5, hold <= 7.5%, edge >= dyn_floor_1b
        5. Tier 2: High/Medium conf, Elite/Strong/Moderate quality,
           edge >= 1.0%, edge_z >= 1.0 (or unavailable), books >= 4
        6. Tier 3: positive-edge plays that don't meet higher tiers
           but also don't trigger Stay Away conditions

    Parameters
    ----------
    entry:
        A dict with at least: edge_pct, confidence, quality_tier,
        quality_score, market_volatility_sigma, edge_z, books_used,
        oldest_update_age_min, market_unstable, market_hold_median.
    settings:
        Reserved for future use.

    Returns
    -------
    dict with keys:
        tier: ``"tier1a"`` | ``"tier1b"`` | ``"tier2"`` | ``"tier3"``
              | ``"avoid"``
        reasons: list[str]  (empty except for avoid)
        dynamic_edge_floor: float  (Tier 1A floor, for debugging)
    """
    settings = settings or {}

    edge = entry.get("edge_pct", 0.0)
    confidence = entry.get("confidence", "")
    quality_tier = entry.get("quality_tier", "")
    sigma = entry.get("market_volatility_sigma", 0.0)
    edge_z = entry.get("edge_z", 0.0)
    market_unstable = entry.get("market_unstable", False)
    books_used = entry.get("books_used", 0)
    oldest_age = entry.get("oldest_update_age_min", 0.0)
    hold_median = entry.get("market_hold_median", 0.0)

    floor_1a = dyn_floor_1a(sigma)

    # ── Hard disqualifiers (block ALL tiers → Stay Away) ──────────
    hard_reasons: list[str] = []
    if market_unstable:
        hard_reasons.append("Unstable market (too many outlier books filtered)")
    if books_used and books_used < _MIN_BOOKS:
        hard_reasons.append(f"Too few books (<{_MIN_BOOKS})")
    if oldest_age > _STALE_THRESHOLD_MIN:
        hard_reasons.append(
            f"Stale lines (oldest update > {_STALE_THRESHOLD_MIN:.0f} min)"
        )
    if edge >= _EDGE_OUTLIER_THRESHOLD and confidence == "Low":
        hard_reasons.append("Edge outlier with low confidence (possible bad data)")

    if hard_reasons:
        _add_market_quality_flags(hard_reasons, entry)
        return {
            "tier": "avoid",
            "reasons": hard_reasons,
            "dynamic_edge_floor": floor_1a,
        }

    # ── Hold gate → Stay Away ─────────────────────────────────────
    if hold_median >= _STAY_AWAY_HOLD_MAX:
        reasons = [
            f"Market hold too high "
            f"({hold_median:.1f}% >= {_STAY_AWAY_HOLD_MAX:.0f}%)"
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
        and books_used >= _TIER1A_BOOKS_MIN
        and hold_median <= _TIER1A_HOLD_MAX
    ):
        return {"tier": "tier1a", "reasons": [], "dynamic_edge_floor": floor_1a}

    # ── Tier 1B (Standard / Aggressive) ───────────────────────────
    floor_1b = dyn_floor_1b(sigma)
    if (
        confidence in ("High", "Medium")
        and quality_tier in ("Elite", "Strong", "Moderate")
        and edge >= floor_1b
        and books_used >= _TIER1B_BOOKS_MIN
        and hold_median <= _TIER1B_HOLD_MAX
    ):
        return {"tier": "tier1b", "reasons": [], "dynamic_edge_floor": floor_1a}

    # ── Tier 2 ────────────────────────────────────────────────────
    t2_conf = confidence in ("High", "Medium")
    t2_qt = quality_tier in ("Elite", "Strong", "Moderate")
    t2_edge = edge >= _TIER2_EDGE
    # edge_z == 0.0 means unavailable (default) — skip the check
    t2_ez = edge_z >= _TIER2_EDGE_Z_MIN if edge_z else True
    t2_books = books_used >= _MIN_BOOKS

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
                "edge_pct": rec.edge_pct,
                "confidence": rec.confidence,
                "quality_score": rec.quality_score,
                "quality_tier": rec.quality_tier,
                "market_volatility_sigma": rec.market_volatility_sigma,
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
            classification = classify_rec(entry, settings=settings)
            entry["tier"] = classification["tier"]
            entry["avoid_reasons"] = classification["reasons"]
            entry["dynamic_edge_floor"] = classification["dynamic_edge_floor"]
            entry["avoid_score"] = _avoid_score(entry)
            entry["distance_to_1b"] = compute_distance_to_1b(entry)

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
        e.setdefault("distance_to_1b", compute_distance_to_1b(e))
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

    return result

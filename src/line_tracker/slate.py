"""Daily Slate aggregator – surfaces the best bets across all events."""

from __future__ import annotations

import os
import statistics
from collections import Counter

from line_tracker.best_bets import recommend_best_bets
from line_tracker.models import BettingLine

# ── tier thresholds ────────────────────────────────────────────────
_TIER1_BASE_EDGE = 3.0
_TIER1_SIGMA_MULT = 1.2
_TIER2_EDGE = 1.5
_TIER2_EDGE_Z_MIN = 1.0
_MIN_BOOKS = 4
_STALE_THRESHOLD_MIN = 120.0
_EDGE_OUTLIER_THRESHOLD = 4.0
_STAY_AWAY_LIMIT = 15

# ── relaxed Tier 2 display thresholds (used when strict mode OFF) ─────
_RELAXED_TIER2_EDGE = 0.5
_RELAXED_TIER2_EDGE_Z_MIN = 0.75


def _slate_score(quality_score: float, edge_pct: float) -> float:
    """0.6 * quality_score + 0.4 * min(edge_pct, 5) * 20"""
    return 0.6 * quality_score + 0.4 * min(edge_pct, 5) * 20


# ── classification ─────────────────────────────────────────────────


def classify_rec(entry: dict, settings: dict | None = None) -> dict:
    """Classify a single slate entry into tier1, tier2, or avoid.

    Runs for **every** recommendation — nothing is skipped before
    classification.

    Parameters
    ----------
    entry:
        A dict with at least: edge_pct, confidence, quality_tier,
        quality_score, market_volatility_sigma, edge_z, books_used,
        oldest_update_age_min, market_unstable.
    settings:
        Reserved for future per-user threshold overrides.

    Returns
    -------
    dict with keys:
        tier:  ``"tier1"`` | ``"tier2"`` | ``"avoid"``
        reasons: list[str]  (empty for tier1/tier2; populated for avoid)
        dynamic_edge_floor: float  (Tier 1 floor used, for debugging)
    """
    edge = entry.get("edge_pct", 0.0)
    confidence = entry.get("confidence", "")
    quality_tier = entry.get("quality_tier", "")
    sigma = entry.get("market_volatility_sigma", 0.0)
    edge_z = entry.get("edge_z", 0.0)
    market_unstable = entry.get("market_unstable", False)
    books_used = entry.get("books_used", 0)
    oldest_age = entry.get("oldest_update_age_min", 0.0)

    dyn_floor = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma

    # ── Hard disqualifiers (block ALL tiers) ──────────────────────
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
        return {
            "tier": "avoid",
            "reasons": hard_reasons,
            "dynamic_edge_floor": dyn_floor,
        }

    # ── Tier 1 ────────────────────────────────────────────────────
    if (
        confidence == "High"
        and quality_tier in ("Elite", "Strong")
        and edge >= dyn_floor
        and edge > 0
    ):
        return {"tier": "tier1", "reasons": [], "dynamic_edge_floor": dyn_floor}

    # ── Tier 2 (independent criteria, NOT Tier 1 lite) ────────────
    t2_conf = confidence in ("High", "Medium")
    t2_qt = quality_tier in ("Elite", "Strong", "Moderate")
    t2_edge = edge >= _TIER2_EDGE
    # edge_z == 0.0 means unavailable (default) — skip the check
    t2_ez = edge_z >= _TIER2_EDGE_Z_MIN if edge_z else True
    t2_pos = edge > 0

    if t2_conf and t2_qt and t2_edge and t2_ez and t2_pos:
        return {"tier": "tier2", "reasons": [], "dynamic_edge_floor": dyn_floor}

    # ── Stay Away — collect human-readable reasons ────────────────
    reasons: list[str] = []

    if confidence == "Low":
        reasons.append(f"Confidence {confidence}")
    elif confidence not in ("High", "Medium"):
        reasons.append(f"Confidence {confidence}")

    if quality_tier not in ("Elite", "Strong", "Moderate"):
        reasons.append(f"Quality tier {quality_tier}")

    if edge <= 0:
        reasons.append(f"Edge not positive ({edge:.1f}%)")
    elif edge < _TIER2_EDGE:
        reasons.append(f"Edge too small ({edge:.1f}% < {_TIER2_EDGE}%)")

    if edge_z and edge_z < _TIER2_EDGE_Z_MIN:
        reasons.append(
            f"Edge Z-score too low ({edge_z:.2f} < {_TIER2_EDGE_Z_MIN})"
        )

    # Note when the rec specifically failed the Tier 1 dynamic floor
    if (
        confidence == "High"
        and quality_tier in ("Elite", "Strong")
        and 0 < edge < dyn_floor
    ):
        reasons.append(
            f"Fails Tier 1 dynamic floor: needs >= {dyn_floor:.1f}% "
            f"given volatility σ={sigma:.3f}"
        )

    if not reasons:
        reasons.append("Does not meet Tier 2 criteria")

    return {"tier": "avoid", "reasons": reasons, "dynamic_edge_floor": dyn_floor}


# ── Stay Away ranking ──────────────────────────────────────────────


def _stay_away_sort_key(entry: dict) -> tuple:
    """Composite key: worst entries sort first (ascending)."""
    conf_order = {"Low": 0, "Medium": 1, "High": 2}
    return (
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
    n_tier1 = counts_by_tier.get("tier1", 0)
    n_tier2 = counts_by_tier.get("tier2", 0)
    n_avoid = counts_by_tier.get("avoid", 0)

    if total > 0 and n_tier1 + n_tier2 + n_avoid == 0:
        warnings.append(
            "total_recs > 0 but Tier1+Tier2+StayAway == 0 "
            "(entries may be lost)"
        )
    if total > 0 and n_avoid == 0:
        warnings.append(
            "total_recs > 0 but StayAway == 0 "
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
) -> dict:
    """Build the daily slate from lines grouped by event.

    Every recommendation is classified **first** (tier1 / tier2 / avoid).
    User display-filters (min_edge, min_quality, hide_low_confidence, etc.)
    are applied only to tier1/tier2 lists afterwards — Stay Away is always
    populated when recs exist.

    Parameters
    ----------
    lines_by_event:
        Mapping of *event_id* → list of ``BettingLine`` objects.
    filters:
        Optional display-filter dict (see ``_passes_filters``).
        ``max_per_event`` (int, default 1) limits recs taken per event.

    Returns
    -------
    dict with keys ``"tier1"``, ``"tier2"``, ``"stay_away"`` (lists of
    slate-entry dicts), ``"counts"`` (always present), ``"debug_stats"``
    (gate-failure counts, sigma/floor stats, warnings — always present),
    and ``"debug"`` (extended counters, present when
    ``LINE_TRACKER_DEBUG`` env-var is set or ``debug`` filter flag is
    True).
    """
    filters = filters or {}
    max_per_event: int = filters.get("max_per_event", 1)

    all_entries: list[dict] = []

    for event_id, lines in lines_by_event.items():
        if not lines:
            continue

        recs = recommend_best_bets(lines)
        if not recs:
            continue

        top_recs = recs[: max(1, max_per_event)]

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
            }

            # Classify — runs for EVERY rec, no pre-filtering
            classification = classify_rec(entry)
            entry["tier"] = classification["tier"]
            entry["avoid_reasons"] = classification["reasons"]
            entry["dynamic_edge_floor"] = classification["dynamic_edge_floor"]

            all_entries.append(entry)

    # Sort by slate_score descending
    all_entries.sort(key=lambda e: e["slate_score"], reverse=True)

    # ── Bucket into tiers (display-filters on tier1/tier2 only) ───
    result: dict = {"tier1": [], "tier2": [], "stay_away": []}
    display_filters = {k: v for k, v in filters.items() if k != "max_per_event"}

    for entry in all_entries:
        if entry["tier"] == "tier1":
            if _passes_filters(entry, display_filters):
                result["tier1"].append(entry)
        elif entry["tier"] == "tier2":
            if _passes_filters(entry, display_filters):
                result["tier2"].append(entry)
        else:
            # Stay Away: only apply markets filter, not edge/quality/confidence
            markets_filter = {}
            if "markets" in filters:
                markets_filter["markets"] = filters["markets"]
            if _passes_filters(entry, markets_filter):
                result["stay_away"].append(entry)

    # Rank Stay Away by "worst-ness" and cap at _STAY_AWAY_LIMIT
    result["stay_away"].sort(key=_stay_away_sort_key)
    result["stay_away"] = result["stay_away"][:_STAY_AWAY_LIMIT]

    # ── Counts (always available) ─────────────────────────────────
    result["counts"] = {
        "total_recs": len(all_entries),
        "tier1": sum(1 for e in all_entries if e["tier"] == "tier1"),
        "tier2": sum(1 for e in all_entries if e["tier"] == "tier2"),
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
            "tier1_count": result["counts"]["tier1"],
            "tier2_count": result["counts"]["tier2"],
            "stay_away_count": result["counts"]["stay_away"],
            "by_confidence": dict(
                Counter(e.get("confidence", "") for e in all_entries)
            ),
            "by_quality_tier": dict(
                Counter(e.get("quality_tier", "") for e in all_entries)
            ),
        }

    return result

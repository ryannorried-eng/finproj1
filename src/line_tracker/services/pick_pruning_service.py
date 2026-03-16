"""Step 2 – Aggressive pick pruning with tight gates.

Only entries that survive ALL gates are promoted to "actionable picks."
Volume comes from frequent snapshots (Step 5), not from relaxing thresholds.
"""

from __future__ import annotations

from line_tracker.config import (
    get_debug_prune_profile,
    get_prune_allowed_alpha_labels,
    get_prune_allowed_tiers,
    get_prune_debug_mode,
    get_prune_max_hold,
    get_prune_min_books,
    get_prune_min_edge_z,
    get_prune_min_ev_shrunk,
    get_prune_min_quality,
)
from line_tracker.services.clv_selection_service import passes_clv_filter


def _resolve_thresholds(
    *,
    allowed_tiers: frozenset[str] | None,
    allowed_alpha: frozenset[str] | None,
    min_edge_z: float | None,
    min_ev_shrunk: float | None,
    min_quality_score: int | None,
    min_books_used: int | None,
    max_market_hold: float | None,
) -> tuple:
    """Resolve gate thresholds from explicit kwargs or config defaults.

    When ``PRUNE_DEBUG_MODE`` is enabled and no explicit kwarg overrides
    are provided, uses relaxed thresholds from ``get_debug_prune_profile``
    so that Tier 3 and Weak-alpha entries survive pruning for diagnostics.
    """
    debug = get_prune_debug_mode()

    if debug:
        profile = get_debug_prune_profile()
        if allowed_tiers is None:
            allowed_tiers = profile["allowed_tiers"]
        if allowed_alpha is None:
            allowed_alpha = profile["allowed_alpha"]
        if min_edge_z is None:
            min_edge_z = profile["min_edge_z"]
        if min_ev_shrunk is None:
            min_ev_shrunk = profile["min_ev_shrunk"]
        if min_quality_score is None:
            min_quality_score = profile["min_quality"]
        if min_books_used is None:
            min_books_used = profile["min_books"]
        if max_market_hold is None:
            max_market_hold = profile["max_hold"]
    else:
        if allowed_tiers is None:
            allowed_tiers = get_prune_allowed_tiers()
        if allowed_alpha is None:
            allowed_alpha = get_prune_allowed_alpha_labels()
        if min_edge_z is None:
            min_edge_z = get_prune_min_edge_z()
        if min_ev_shrunk is None:
            min_ev_shrunk = get_prune_min_ev_shrunk()
        if min_quality_score is None:
            min_quality_score = get_prune_min_quality()
        if min_books_used is None:
            min_books_used = get_prune_min_books()
        if max_market_hold is None:
            max_market_hold = get_prune_max_hold()

    return (
        allowed_tiers, allowed_alpha, min_edge_z, min_ev_shrunk,
        min_quality_score, min_books_used, max_market_hold,
    )


def prune_picks(
    entries: list[dict],
    *,
    clv_profile: dict | None = None,
    allowed_tiers: frozenset[str] | None = None,
    allowed_alpha: frozenset[str] | None = None,
    min_edge_z: float | None = None,
    min_ev_shrunk: float | None = None,
    min_quality_score: int | None = None,
    min_books_used: int | None = None,
    max_market_hold: float | None = None,
) -> list[dict]:
    """Apply a strict gate cascade and return only surviving entries.

    Gate order:
    1. tier in allowed_tiers
    2. alpha_label in allowed_alpha
    3. edge_z >= min_edge_z
    4. edge_ev_shrunk > min_ev_shrunk
    5. quality_score >= min_quality_score
    6. books_used >= min_books_used
    7. market_hold_median <= max_market_hold
    8. CLV filter (if profile provided)
    """
    survivors, _ = prune_picks_with_reasons(
        entries,
        clv_profile=clv_profile,
        allowed_tiers=allowed_tiers,
        allowed_alpha=allowed_alpha,
        min_edge_z=min_edge_z,
        min_ev_shrunk=min_ev_shrunk,
        min_quality_score=min_quality_score,
        min_books_used=min_books_used,
        max_market_hold=max_market_hold,
    )
    return survivors


def prune_picks_with_reasons(
    entries: list[dict],
    *,
    clv_profile: dict | None = None,
    allowed_tiers: frozenset[str] | None = None,
    allowed_alpha: frozenset[str] | None = None,
    min_edge_z: float | None = None,
    min_ev_shrunk: float | None = None,
    min_quality_score: int | None = None,
    min_books_used: int | None = None,
    max_market_hold: float | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Apply the same gate cascade as ``prune_picks`` and also return a
    histogram of failure counts per gate.

    Returns
    -------
    (survivors, reasons_histogram)
        *survivors* is the list of entries passing all gates.
        *reasons_histogram* maps gate key to the number of entries that
        failed at that gate (first failure only — each entry is counted
        once).  Keys: tier, alpha, edge_z, ev_shrunk, quality, books,
        hold, clv.
    """
    (
        allowed_tiers, allowed_alpha, min_edge_z, min_ev_shrunk,
        min_quality_score, min_books_used, max_market_hold,
    ) = _resolve_thresholds(
        allowed_tiers=allowed_tiers,
        allowed_alpha=allowed_alpha,
        min_edge_z=min_edge_z,
        min_ev_shrunk=min_ev_shrunk,
        min_quality_score=min_quality_score,
        min_books_used=min_books_used,
        max_market_hold=max_market_hold,
    )

    reasons: dict[str, int] = {
        "tier": 0,
        "alpha": 0,
        "edge_z": 0,
        "ev_shrunk": 0,
        "quality": 0,
        "books": 0,
        "hold": 0,
        "clv": 0,
    }

    result: list[dict] = []
    for entry in entries:
        if entry.get("tier", "") not in allowed_tiers:
            reasons["tier"] += 1
            continue
        alpha = (entry.get("alpha_label") or "").strip().title()
        if alpha not in allowed_alpha:
            reasons["alpha"] += 1
            continue
        if (entry.get("edge_z") or 0) < min_edge_z:
            reasons["edge_z"] += 1
            continue
        if (entry.get("edge_ev_shrunk") or 0) <= min_ev_shrunk:
            reasons["ev_shrunk"] += 1
            continue
        if (entry.get("quality_score") or 0) < min_quality_score:
            reasons["quality"] += 1
            continue
        if (entry.get("books_used") or 0) < min_books_used:
            reasons["books"] += 1
            continue
        if (entry.get("market_hold_median") or 0) > max_market_hold:
            reasons["hold"] += 1
            continue
        if clv_profile and not passes_clv_filter(entry, clv_profile):
            reasons["clv"] += 1
            continue
        result.append(entry)
    return result, reasons

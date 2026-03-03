"""Step 2 – Aggressive pick pruning with tight gates.

Only entries that survive ALL gates are promoted to "actionable picks."
Volume comes from frequent snapshots (Step 5), not from relaxing thresholds.
"""

from __future__ import annotations

from line_tracker.config import (
    get_prune_allowed_alpha_labels,
    get_prune_allowed_tiers,
    get_prune_max_hold,
    get_prune_min_books,
    get_prune_min_edge_z,
    get_prune_min_ev_shrunk,
    get_prune_min_quality,
)
from line_tracker.services.clv_selection_service import passes_clv_filter


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

    result: list[dict] = []
    for entry in entries:
        if entry.get("tier", "") not in allowed_tiers:
            continue
        if entry.get("alpha_label", "") not in allowed_alpha:
            continue
        if (entry.get("edge_z") or 0) < min_edge_z:
            continue
        if (entry.get("edge_ev_shrunk") or 0) <= min_ev_shrunk:
            continue
        if (entry.get("quality_score") or 0) < min_quality_score:
            continue
        if (entry.get("books_used") or 0) < min_books_used:
            continue
        if (entry.get("market_hold_median") or 0) > max_market_hold:
            continue
        if clv_profile and not passes_clv_filter(entry, clv_profile):
            continue
        result.append(entry)
    return result

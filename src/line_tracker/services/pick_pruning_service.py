"""Step 2 – Aggressive pick pruning with tight gates.

Only entries that survive ALL gates are promoted to "actionable picks."
Volume comes from frequent snapshots (Step 5), not from relaxing thresholds.
"""

from __future__ import annotations

from line_tracker.services.clv_selection_service import passes_clv_filter

# ── Default gate thresholds ───────────────────────────────────────────
ALLOWED_TIERS = frozenset({"tier1a", "tier1b", "tier2"})
ALLOWED_ALPHA = frozenset({"Strong", "Neutral"})
MIN_EDGE_Z = 1.75
MIN_QUALITY_SCORE = 65
MIN_BOOKS_USED = 5
MAX_MARKET_HOLD = 7.0


def prune_picks(
    entries: list[dict],
    *,
    clv_profile: dict | None = None,
    allowed_tiers: frozenset[str] = ALLOWED_TIERS,
    allowed_alpha: frozenset[str] = ALLOWED_ALPHA,
    min_edge_z: float = MIN_EDGE_Z,
    min_quality_score: int = MIN_QUALITY_SCORE,
    min_books_used: int = MIN_BOOKS_USED,
    max_market_hold: float = MAX_MARKET_HOLD,
) -> list[dict]:
    """Apply a strict gate cascade and return only surviving entries.

    Gate order:
    1. tier in allowed_tiers
    2. alpha_label in allowed_alpha
    3. edge_z >= min_edge_z
    4. edge_ev_shrunk > 0
    5. quality_score >= min_quality_score
    6. books_used >= min_books_used
    7. market_hold_median <= max_market_hold
    8. CLV filter (if profile provided)
    """
    result: list[dict] = []
    for entry in entries:
        if entry.get("tier", "") not in allowed_tiers:
            continue
        if entry.get("alpha_label", "") not in allowed_alpha:
            continue
        if (entry.get("edge_z") or 0) < min_edge_z:
            continue
        if (entry.get("edge_ev_shrunk") or 0) <= 0:
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

"""Daily Slate aggregator – surfaces the best bets across all events."""

from __future__ import annotations

from line_tracker.best_bets import recommend_best_bets
from line_tracker.models import BettingLine


# ── tier thresholds ────────────────────────────────────────────────
_TIER1_BASE_EDGE = 3.0
_TIER1_SIGMA_MULT = 1.2
_TIER2_QUALITY = 40
_TIER2_EDGE = 1.0
_MIN_BOOKS = 4
_STALE_THRESHOLD_MIN = 120.0
_EDGE_OUTLIER_THRESHOLD = 4.0


def _slate_score(quality_score: float, edge_pct: float) -> float:
    """0.6 * quality_score + 0.4 * min(edge_pct, 5) * 20"""
    return 0.6 * quality_score + 0.4 * min(edge_pct, 5) * 20


def _assign_tier(
    rec_quality: int,
    rec_edge: float,
    rec_confidence: str,
    rec_market_unstable: bool,
    rec_books_used: int = 0,
    rec_oldest_age_min: float = 0.0,
    rec_quality_tier: str = "",
    rec_volatility_sigma: float = 0.0,
) -> tuple[str, list[str]]:
    """Return (tier, avoid_reasons).

    tier1  – edge >= dynamic_min_edge AND confidence == "High"
             AND quality_tier in {"Elite", "Strong"}
             where dynamic_min_edge = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma
    tier2  – quality >= 40 AND edge >= 1.0
    avoid  – everything else (with reasons)
    """
    reasons: list[str] = []

    if rec_market_unstable:
        reasons.append("Unstable market (too many outlier books filtered)")
    if rec_edge < _TIER2_EDGE:
        reasons.append(f"Edge too small ({rec_edge:.2f}% < {_TIER2_EDGE}%)")
    if rec_quality < _TIER2_QUALITY:
        reasons.append(f"Low quality score ({rec_quality} < {_TIER2_QUALITY})")
    if rec_confidence == "Low":
        reasons.append("Low confidence (books disagree)")
    if rec_books_used and rec_books_used < _MIN_BOOKS:
        reasons.append(f"Too few books (<{_MIN_BOOKS})")
    if rec_oldest_age_min > _STALE_THRESHOLD_MIN:
        reasons.append(
            f"Stale lines (oldest update > {_STALE_THRESHOLD_MIN:.0f} min)"
        )
    if rec_edge >= _EDGE_OUTLIER_THRESHOLD and rec_confidence == "Low":
        reasons.append("Edge outlier with low confidence (possible bad data)")

    if reasons:
        return "avoid", reasons

    # Dynamic edge floor for tier1
    dyn_floor = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * rec_volatility_sigma
    if (
        rec_edge >= dyn_floor
        and rec_confidence == "High"
        and rec_quality_tier in ("Elite", "Strong")
    ):
        return "tier1", []

    return "tier2", []


def _passes_filters(entry: dict, filters: dict) -> bool:
    """Return True if the entry survives all user-supplied filters."""
    if filters.get("min_edge") and entry["edge_pct"] < filters["min_edge"]:
        return False
    if filters.get("min_quality") and entry["quality_score"] < filters["min_quality"]:
        return False
    if filters.get("markets") and entry["market"] not in filters["markets"]:
        return False
    if filters.get("hide_low_confidence") and entry["confidence"] == "Low":
        return False
    if filters.get("books_used_min") and entry["books_used"] < filters["books_used_min"]:
        return False
    return True


def build_daily_slate(
    lines_by_event: dict[str, list[BettingLine]],
    *,
    filters: dict | None = None,
) -> dict:
    """Build the daily slate from lines grouped by event.

    Parameters
    ----------
    lines_by_event:
        Mapping of *event_id* → list of ``BettingLine`` objects for that event.
    filters:
        Optional dict with any of the following keys:
        - ``max_per_event``     (int)  – max recommendations per event (default 1)
        - ``min_edge``          (float) – drop entries below this edge_pct
        - ``min_quality``       (int)   – drop entries below this quality_score
        - ``markets``           (list[str]) – only keep these markets
        - ``hide_low_confidence`` (bool) – drop "Low" confidence entries
        - ``books_used_min``    (int)   – drop entries with fewer books

    Returns
    -------
    dict with keys ``"tier1"``, ``"tier2"``, ``"avoid"`` – each a list of
    slate-entry dicts sorted by *slate_score* descending.
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

        # Take top recommendation(s)
        top_recs = recs[: max(1, max_per_event)]

        # Derive shared metadata from the first line of the event
        sample_line = lines[0]
        event_name = sample_line.event
        commence_time = sample_line.commence_time

        for rec in top_recs:
            score = _slate_score(rec.quality_score, rec.edge_pct)
            tier, avoid_reasons = _assign_tier(
                rec.quality_score,
                rec.edge_pct,
                rec.confidence,
                rec.market_unstable,
                rec_books_used=rec.books_used_count,
                rec_oldest_age_min=rec.oldest_update_age_min,
                rec_quality_tier=rec.quality_tier,
                rec_volatility_sigma=rec.market_volatility_sigma,
            )

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
                "slate_score": score,
                "books_used": rec.books_used_count,
                "updated_age_min": rec.newest_update_age_min,
                "oldest_update_age_min": rec.oldest_update_age_min,
                "tier": tier,
                "avoid_reasons": avoid_reasons,
            }

            if _passes_filters(entry, filters):
                all_entries.append(entry)

    # Sort every tier by slate_score descending
    all_entries.sort(key=lambda e: e["slate_score"], reverse=True)

    result: dict[str, list[dict]] = {"tier1": [], "tier2": [], "avoid": []}
    for entry in all_entries:
        result[entry["tier"]].append(entry)

    return result

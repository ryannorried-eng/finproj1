"""Explanation service — surfaces BestBetResult data for Debug mode UI."""

from __future__ import annotations

from typing import Any

from line_tracker.best_bets import (
    extract_best_bet_results,
    recommend_best_bets,
)
from line_tracker.core.logging import get_logger
from line_tracker.models import BestBetResult, BettingLine


def get_recommendation_explanations(
    lines_for_event: list[BettingLine],
    *,
    top_n: int = 3,
    include_raw_inputs: bool = False,
) -> list[dict[str, Any]]:
    """Return explanation dicts for the top recommendations.

    This is the primary entry-point for the Debug mode UI.  It runs
    the full recommendation pipeline, extracts ``BestBetResult`` objects,
    and returns one serialisable dict per recommendation containing
    the explanation and key metrics.

    Parameters
    ----------
    lines_for_event:
        All betting lines for a single event.
    top_n:
        Maximum recommendations.
    include_raw_inputs:
        When True, include the ``raw_inputs`` debug payload.

    Returns
    -------
    list[dict[str, Any]]
        One dict per recommendation with keys: ``market``, ``selection``,
        ``side``, ``line``, ``edge_pct``, ``consensus_prob``,
        ``best_odds_american``, ``best_odds_decimal``, ``confidence``,
        ``quality_score``, ``quality_tier``, ``explanation``,
        and optionally ``raw_inputs``.
    """
    log = get_logger(__name__, source="explanation_service")

    recs = recommend_best_bets(lines_for_event, top_n=top_n)
    results = extract_best_bet_results(recs)

    log.info(
        "get_recommendation_explanations recs=%d results=%d",
        len(recs),
        len(results),
    )

    out: list[dict[str, Any]] = []
    for bbr in results:
        entry: dict[str, Any] = {
            "market": bbr.market,
            "selection": bbr.selection,
            "side": bbr.side,
            "line": bbr.line,
            "edge_pct": bbr.edge_pct,
            "consensus_prob": bbr.consensus_prob,
            "best_odds_american": bbr.best_odds_american,
            "best_odds_decimal": bbr.best_odds_decimal,
            "best_sportsbook": bbr.best_sportsbook,
            "confidence": bbr.confidence,
            "quality_score": bbr.quality_score,
            "quality_tier": bbr.quality_tier,
            "books_used": bbr.books_used,
            "books_used_count": bbr.books_used_count,
            "volatility_sigma": bbr.volatility_sigma,
            "recency_weight": bbr.recency_weight,
            "outliers_removed": bbr.outliers_removed,
            "edge_z": bbr.edge_z,
            "edge_ev": bbr.edge_ev,
            "edge_ev_shrunk": bbr.edge_ev_shrunk,
            "ev_roi": bbr.ev_roi,
            "ev_100": bbr.ev_100,
            "kelly_suggested": bbr.kelly_suggested,
            "sizing_note": bbr.sizing_note,
            "explanation": bbr.explanation,
        }
        if include_raw_inputs:
            entry["raw_inputs"] = bbr.raw_inputs
        out.append(entry)

    return out


def get_slate_explanations(
    lines_by_event: dict[str, list[BettingLine]],
    *,
    top_n: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Return explanation dicts keyed by event ID.

    Iterates over all events in the slate and returns explanations
    for the top recommendations in each.
    """
    log = get_logger(__name__, source="explanation_service")
    result: dict[str, list[dict[str, Any]]] = {}

    for event_id, lines in lines_by_event.items():
        if not lines:
            continue
        explanations = get_recommendation_explanations(
            lines, top_n=top_n,
        )
        if explanations:
            result[event_id] = explanations

    log.info(
        "get_slate_explanations events=%d events_with_explanations=%d",
        len(lines_by_event),
        len(result),
    )
    return result


def format_explanation_summary(bbr: BestBetResult) -> str:
    """Format a human-readable explanation summary for a single result.

    Useful for CLI output or logging.
    """
    expl = bbr.explanation
    lines = [
        f"{bbr.market.upper()} {bbr.selection} ({bbr.side})",
        f"  Edge: {bbr.edge_pct:.2f}% | Consensus: {bbr.consensus_prob:.4f}",
        f"  Best odds: {bbr.best_odds_american:+.0f} ({bbr.best_odds_decimal:.3f})",
        f"  Confidence: {bbr.confidence} | Quality: "
        f"{bbr.quality_tier} ({bbr.quality_score})",
    ]

    # Edge breakdown
    eb = expl.get("edge_breakdown", {})
    if eb:
        lines.append(
            f"  Breakeven: {eb.get('breakeven_prob', 0):.4f} "
            f"| Edge PP: {eb.get('edge_pp', 0):.4f}"
        )

    # Consensus method
    cm = expl.get("consensus_method", "")
    if cm:
        lines.append(f"  Consensus method: {cm}")

    # Quality factors
    qf = expl.get("quality_factors", {})
    if qf:
        lines.append(
            f"  Quality breakdown: edge={qf.get('edge_score', 0):.0f} "
            f"agree={qf.get('agreement_score', 0):.0f} "
            f"cover={qf.get('coverage_score', 0):.0f} "
            f"fresh={qf.get('freshness_score', 0):.0f}"
        )

    # Outlier info
    oi = expl.get("outlier_info", {})
    if oi.get("outlier_filtered"):
        lines.append(
            f"  Outliers: {oi.get('outliers_removed', 0)} removed "
            f"(rate={oi.get('outlier_rate', 0):.2%})"
        )

    # Kelly
    ki = expl.get("kelly", {})
    if ki:
        lines.append(f"  Kelly: {ki.get('sizing_note', '')}")

    return "\n".join(lines)

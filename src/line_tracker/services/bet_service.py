"""Bet persistence and settlement orchestration service."""

from __future__ import annotations

from line_tracker.bet_history import (
    close_bet_clv,
    persist_bet_with_snapshot,
    settle_bet_persistent,
)
from line_tracker.core.logging import get_logger
from line_tracker.core.math import american_to_decimal, build_recommendation_id


def _get(obj, key, default=None):
    """Read *key* from a dict or attribute of *obj*."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def submit_bet(
    bet,
    store,
    *,
    source_page: str | None = None,
    recommendation=None,
    rank: int | None = None,
) -> str:
    """Persist bet and CLV pick snapshot atomically. Returns bet_id.

    Parameters
    ----------
    bet : Bet
        The bet to persist.
    store : LineStore
        Database connection wrapper.
    source_page : str, optional
        ``"slate"`` or ``"shopping"`` — the page the bet was placed from.
    recommendation : BetRecommendation | dict, optional
        The recommendation object (or dict) that the bet was based on.
        When provided, ``recommendation_id`` and ``execution_delta_decimal``
        are computed and persisted.
    rank : int, optional
        The rank of the recommendation in its list at the time of pick.
    """
    log = get_logger(__name__, source="bet_service")

    rec_meta: dict | None = None
    if recommendation is not None:
        rec_id = build_recommendation_id(recommendation)
        consensus_prob = _get(recommendation, "consensus_prob", 0.0)
        best_odds = _get(recommendation, "best_odds", 0)
        best_odds_decimal = american_to_decimal(best_odds) if best_odds else 0.0
        fair_decimal = (1.0 / consensus_prob) if consensus_prob > 0 else 0.0
        execution_delta = (
            round(best_odds_decimal - fair_decimal, 6) if fair_decimal > 0 else None
        )

        rec_meta = {
            "source_page": source_page,
            "recommendation_id": rec_id,
            "rank_at_pick": rank,
            "quality_tier_at_pick": _get(recommendation, "quality_tier"),
            "edge_pct_at_pick": _get(recommendation, "edge_pct"),
            "consensus_prob_at_pick": consensus_prob,
            "execution_delta_decimal": execution_delta,
        }
    elif source_page is not None:
        rec_meta = {"source_page": source_page}

    bet_id = persist_bet_with_snapshot(bet, store, recommendation_meta=rec_meta)
    legs_count = len(getattr(bet, "legs", []) or [])
    stake = getattr(bet, "stake", None)
    log.info(
        "submit_bet persisted bet_id=%s legs=%s stake=%s source=%s",
        bet_id, legs_count, stake, source_page,
    )
    return bet_id


def settle_bet(bet_id: str, outcome: str, store) -> None:
    """Close CLV then persist settled outcome for a placed bet."""
    log = get_logger(__name__, source="bet_service")
    before = getattr(
        getattr(store, "_conn", None), "total_changes", None,
    )
    close_bet_clv(bet_id, store)
    settle_bet_persistent(bet_id, outcome, store)
    after = getattr(
        getattr(store, "_conn", None), "total_changes", None,
    )
    changed = (
        (after - before)
        if (before is not None and after is not None)
        else "unknown"
    )
    log.info(
        "settle_bet persisted bet_id=%s outcome=%s rows_updated=%s",
        bet_id, outcome, changed,
    )

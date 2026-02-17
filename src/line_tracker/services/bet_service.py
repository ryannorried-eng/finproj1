"""Bet persistence and settlement orchestration service."""

from __future__ import annotations

from line_tracker.bet_history import (
    close_bet_clv,
    persist_bet_with_snapshot,
    settle_bet_persistent,
)
from line_tracker.core.logging import get_logger


def submit_bet(bet, store) -> str:
    """Persist bet and CLV pick snapshot atomically. Returns bet_id."""
    log = get_logger(__name__, source="bet_service")
    bet_id = persist_bet_with_snapshot(bet, store)
    legs_count = len(getattr(bet, "legs", []) or [])
    stake = getattr(bet, "stake", None)
    log.info(
        "submit_bet persisted bet_id=%s legs=%s stake=%s",
        bet_id, legs_count, stake,
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

"""Bet persistence and settlement orchestration service."""

from __future__ import annotations

from line_tracker.bet_history import close_bet_clv, persist_bet_with_snapshot, settle_bet_persistent


def submit_bet(bet, store) -> str:
    """Persist bet and CLV pick snapshot atomically. Returns bet_id."""
    return persist_bet_with_snapshot(bet, store)


def settle_bet(bet_id: str, outcome: str, store) -> None:
    """Close CLV then persist settled outcome for a placed bet."""
    close_bet_clv(bet_id, store)
    settle_bet_persistent(bet_id, outcome, store)

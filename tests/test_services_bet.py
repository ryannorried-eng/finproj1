"""Tests for bet service orchestration."""

from line_tracker.services import bet_service


class _Store:
    pass


def test_submit_bet_calls_persist_and_returns_id(monkeypatch):
    calls = {}

    def _fake_persist(bet, store):
        calls["bet"] = bet
        calls["store"] = store
        return "bet-123"

    monkeypatch.setattr(bet_service, "persist_bet_with_snapshot", _fake_persist)

    store = _Store()
    bet = object()
    bet_id = bet_service.submit_bet(bet, store)

    assert bet_id == "bet-123"
    assert calls["bet"] is bet
    assert calls["store"] is store


def test_settle_bet_calls_close_then_settle(monkeypatch):
    events = []

    monkeypatch.setattr(
        bet_service,
        "close_bet_clv",
        lambda bet_id, store: events.append(("close", bet_id, store)),
    )
    monkeypatch.setattr(
        bet_service,
        "settle_bet_persistent",
        lambda bid, out, s: events.append(("settle", bid, out, s)),
    )

    store = _Store()
    bet_service.settle_bet("b1", "won", store)

    assert events == [
        ("close", "b1", store),
        ("settle", "b1", "won", store),
    ]

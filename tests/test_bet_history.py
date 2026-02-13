"""Tests for bet history data layer."""

import pytest

from line_tracker.bet_history import (
    Bet,
    create_bet,
    delete_bet,
    init_bet_state,
    settle_bet,
    submit_bet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leg(**overrides):
    base = {
        "sport": "NFL",
        "event_name": "Bills @ Chiefs",
        "sportsbook": "DraftKings",
        "market": "ML",
        "selection": "Home",
        "line": None,
        "odds": -150,
        "fetched_at": "2026-01-19T18:00:00",
    }
    base.update(overrides)
    return base


def _slip_state(legs=None, book="DraftKings", stake=100.0):
    """Return a dict mimicking Streamlit session_state with a filled slip."""
    if legs is None:
        legs = [_leg()]
    return {
        "bet_slip": list(legs),
        "slip_book": book,
        "slip_stake": stake,
    }


# ---------------------------------------------------------------------------
# create_bet
# ---------------------------------------------------------------------------

class TestCreateBet:
    def test_straight_bet(self):
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg(odds=-150)])
        assert isinstance(bet, Bet)
        assert bet.sportsbook == "DK"
        assert bet.stake == 100
        assert bet.status == "active"
        assert bet.settled_at is None
        # -150 → decimal 1.6667, profit ~66.67
        assert bet.combined_decimal == pytest.approx(1.6667, abs=0.001)
        assert bet.profit == pytest.approx(66.67, abs=0.01)
        assert bet.total_payout == pytest.approx(166.67, abs=0.01)

    def test_parlay_bet(self):
        legs = [_leg(odds=150), _leg(odds=-110, event_name="Eagles @ Cowboys")]
        bet = create_bet(stake=100, sportsbook="DK", legs=legs)
        # +150 → 2.5, -110 → ~1.909, combined ~4.77
        assert bet.combined_decimal > 4.7
        assert bet.profit > 370
        assert len(bet.legs) == 2

    def test_unique_id(self):
        a = create_bet(stake=100, sportsbook="DK", legs=[_leg()])
        b = create_bet(stake=100, sportsbook="DK", legs=[_leg()])
        assert a.id != b.id

    def test_empty_legs_raises(self):
        with pytest.raises(ValueError, match="at least one leg"):
            create_bet(stake=100, sportsbook="DK", legs=[])

    def test_zero_stake_raises(self):
        with pytest.raises(ValueError, match="positive"):
            create_bet(stake=0, sportsbook="DK", legs=[_leg()])

    def test_negative_stake_raises(self):
        with pytest.raises(ValueError, match="positive"):
            create_bet(stake=-50, sportsbook="DK", legs=[_leg()])

    def test_legs_are_copied(self):
        original = [_leg()]
        bet = create_bet(stake=100, sportsbook="DK", legs=original)
        original.append(_leg())
        assert len(bet.legs) == 1

    def test_created_at_set(self):
        bet = create_bet(stake=100, sportsbook="DK", legs=[_leg()])
        assert bet.created_at  # non-empty ISO string


# ---------------------------------------------------------------------------
# init_bet_state
# ---------------------------------------------------------------------------

class TestInitBetState:
    def test_creates_lists_when_missing(self):
        state: dict = {}
        init_bet_state(state)
        assert state["active_bets"] == []
        assert state["settled_bets"] == []

    def test_preserves_existing(self):
        sentinel = [Bet(
            id="x", sportsbook="DK", legs=[], combined_decimal=1.0,
            combined_american=-10000, stake=10, profit=0, total_payout=10,
            status="active", created_at="t",
        )]
        state: dict = {"active_bets": sentinel}
        init_bet_state(state)
        assert state["active_bets"] is sentinel
        assert state["settled_bets"] == []


# ---------------------------------------------------------------------------
# submit_bet
# ---------------------------------------------------------------------------

class TestSubmitBet:
    def test_basic_submit(self):
        state = _slip_state()
        bet = submit_bet(state, stake=100)
        assert bet.status == "active"
        assert len(state["active_bets"]) == 1
        assert state["active_bets"][0] is bet
        # Slip should be cleared
        assert state["bet_slip"] == []
        assert state["slip_book"] is None
        assert state["slip_stake"] == 0.0

    def test_empty_slip_raises(self):
        state = _slip_state(legs=[])
        with pytest.raises(ValueError, match="empty"):
            submit_bet(state, stake=100)

    def test_no_book_raises(self):
        state = _slip_state(book=None)
        with pytest.raises(ValueError, match="sportsbook"):
            submit_bet(state, stake=100)

    def test_multiple_submits_accumulate(self):
        state = _slip_state()
        submit_bet(state, stake=50)
        # Refill the slip
        state["bet_slip"] = [_leg()]
        state["slip_book"] = "DraftKings"
        submit_bet(state, stake=75)
        assert len(state["active_bets"]) == 2


# ---------------------------------------------------------------------------
# settle_bet
# ---------------------------------------------------------------------------

class TestSettleBet:
    def _state_with_active(self):
        state = _slip_state()
        bet = submit_bet(state, stake=100)
        return state, bet

    def test_settle_won(self):
        state, bet = self._state_with_active()
        original_profit = bet.profit
        original_payout = bet.total_payout
        settled = settle_bet(state, bet.id, "won")
        assert settled.status == "won"
        assert settled.settled_at is not None
        assert len(state["active_bets"]) == 0
        assert len(state["settled_bets"]) == 1
        assert state["settled_bets"][0] is settled
        # Won keeps original profit/payout as floats
        assert settled.profit == pytest.approx(original_profit)
        assert settled.total_payout == pytest.approx(original_payout)
        assert isinstance(settled.profit, float)
        assert isinstance(settled.total_payout, float)

    def test_settle_lost(self):
        state, bet = self._state_with_active()
        settled = settle_bet(state, bet.id, "lost")
        assert settled.status == "lost"
        assert settled.profit == -100.0
        assert settled.total_payout == 0.0
        assert isinstance(settled.profit, float)
        assert isinstance(settled.total_payout, float)

    def test_settle_push(self):
        state, bet = self._state_with_active()
        settled = settle_bet(state, bet.id, "push")
        assert settled.status == "push"
        assert settled.profit == 0.0
        assert settled.total_payout == 100.0
        assert isinstance(settled.profit, float)
        assert isinstance(settled.total_payout, float)

    def test_settle_invalid_outcome(self):
        state, bet = self._state_with_active()
        with pytest.raises(ValueError, match="Invalid outcome"):
            settle_bet(state, bet.id, "void")

    def test_settle_unknown_id(self):
        state, _ = self._state_with_active()
        with pytest.raises(KeyError, match="No active bet"):
            settle_bet(state, "nonexistent", "won")

    def test_settle_idempotent_removal(self):
        state = _slip_state()
        b1 = submit_bet(state, stake=100)
        state["bet_slip"] = [_leg()]
        state["slip_book"] = "DraftKings"
        b2 = submit_bet(state, stake=50)
        settle_bet(state, b1.id, "won")
        assert len(state["active_bets"]) == 1
        assert state["active_bets"][0].id == b2.id


# ---------------------------------------------------------------------------
# delete_bet
# ---------------------------------------------------------------------------

class TestDeleteBet:
    def test_delete_active(self):
        state = _slip_state()
        bet = submit_bet(state, stake=100)
        removed = delete_bet(state, bet.id)
        assert removed.id == bet.id
        assert len(state["active_bets"]) == 0

    def test_delete_settled(self):
        state = _slip_state()
        bet = submit_bet(state, stake=100)
        settle_bet(state, bet.id, "lost")
        removed = delete_bet(state, bet.id)
        assert removed.id == bet.id
        assert len(state["settled_bets"]) == 0

    def test_delete_unknown_raises(self):
        state: dict = {}
        init_bet_state(state)
        with pytest.raises(KeyError, match="No bet"):
            delete_bet(state, "missing")


# ---------------------------------------------------------------------------
# Round-trip / integration
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_submit_settle_delete(self):
        state = _slip_state(legs=[
            _leg(odds=-110),
            _leg(odds=130, event_name="Eagles @ Cowboys"),
        ])
        bet = submit_bet(state, stake=200)
        assert bet.status == "active"
        assert len(bet.legs) == 2

        settled = settle_bet(state, bet.id, "won")
        assert settled.status == "won"

        removed = delete_bet(state, bet.id)
        assert removed.id == bet.id
        assert state["active_bets"] == []
        assert state["settled_bets"] == []

"""Tests for bet slip odds helpers and parlay math."""

from line_tracker.bet_slip import (
    american_profit,
    american_to_decimal,
    american_total_return,
    decimal_to_american,
    format_american,
    has_conflicting_leg,
    implied_prob_from_american,
    is_duplicate_leg,
    parlay_payout,
)

# --- american_to_decimal ---

def test_a2d_positive():
    assert american_to_decimal(150) == 2.5


def test_a2d_negative():
    assert american_to_decimal(-200) == 1.5


def test_a2d_even():
    assert american_to_decimal(100) == 2.0


def test_a2d_zero():
    # +0 treated as even money edge case
    assert american_to_decimal(0) == 1.0


# --- decimal_to_american ---

def test_d2a_above_2():
    assert decimal_to_american(2.5) == 150.0


def test_d2a_below_2():
    assert decimal_to_american(1.5) == -200.0


def test_d2a_exactly_2():
    assert decimal_to_american(2.0) == 100.0


def test_d2a_at_1():
    assert decimal_to_american(1.0) == 0.0


# --- american_profit / total_return ---

def test_profit_positive_odds():
    # +150, stake 100 → profit 150
    assert american_profit(100, 150) == 150.0


def test_profit_negative_odds():
    # -200, stake 100 → profit 50
    assert american_profit(100, -200) == 50.0


def test_total_return_positive_odds():
    # +150, stake 100 → return 250
    assert american_total_return(100, 150) == 250.0


def test_total_return_negative_odds():
    # -200, stake 100 → return 150
    assert american_total_return(100, -200) == 150.0


# --- implied_prob_from_american ---

def test_implied_prob_favorite():
    # -200 → 200 / 300 ≈ 0.6667
    assert round(implied_prob_from_american(-200), 4) == 0.6667


def test_implied_prob_underdog():
    # +150 → 100 / 250 = 0.4
    assert implied_prob_from_american(150) == 0.4


def test_implied_prob_even():
    assert implied_prob_from_american(0) == 0.5


# --- parlay_payout ---

def test_parlay_two_legs():
    result = parlay_payout(100, [150, -110])
    # +150 → 2.50, -110 → 1.909..., combined ≈ 4.7727
    assert result["combined_decimal"] > 4.7
    assert result["profit"] > 370
    assert result["total_return"] == round(result["profit"] + 100, 2)


def test_parlay_single_leg():
    result = parlay_payout(100, [-150])
    assert result["combined_decimal"] == round(100 / 150 + 1, 4)
    assert result["total_return"] == round(100 * result["combined_decimal"], 2)


def test_parlay_three_legs():
    result = parlay_payout(50, [100, 100, 100])
    # +100 → 2.0 each, combined = 8.0
    assert result["combined_decimal"] == 8.0
    assert result["total_return"] == 400.0
    assert result["profit"] == 350.0


# --- format_american ---

def test_format_positive():
    assert format_american(150) == "+150"


def test_format_negative():
    assert format_american(-110) == "-110"


def test_format_even():
    assert format_american(0) == "EVEN"


# --- is_duplicate_leg ---

def _make_leg(**overrides):
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


def test_duplicate_detected():
    leg = _make_leg()
    slip = [_make_leg()]
    assert is_duplicate_leg(slip, leg)


def test_different_odds_not_duplicate():
    leg = _make_leg(odds=-145)
    slip = [_make_leg(odds=-150)]
    assert not is_duplicate_leg(slip, leg)


def test_different_sportsbook_not_duplicate():
    leg = _make_leg(sportsbook="FanDuel")
    slip = [_make_leg(sportsbook="DraftKings")]
    assert not is_duplicate_leg(slip, leg)


# --- has_conflicting_leg ---

def test_conflict_detected():
    leg = _make_leg(selection="Away")
    slip = [_make_leg(selection="Home")]
    assert has_conflicting_leg(slip, leg)


def test_no_conflict_same_selection():
    leg = _make_leg(selection="Home", sportsbook="FanDuel")
    slip = [_make_leg(selection="Home", sportsbook="DraftKings")]
    assert not has_conflicting_leg(slip, leg)


def test_no_conflict_different_event():
    leg = _make_leg(event_name="Eagles @ Cowboys", selection="Away")
    slip = [_make_leg(event_name="Bills @ Chiefs", selection="Home")]
    assert not has_conflicting_leg(slip, leg)


def test_no_conflict_different_market():
    leg = _make_leg(market="Spread", selection="Away")
    slip = [_make_leg(market="ML", selection="Home")]
    assert not has_conflicting_leg(slip, leg)

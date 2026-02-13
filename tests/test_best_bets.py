"""Tests for best_bets module."""

from datetime import datetime

from line_tracker.best_bets import (
    BetRecommendation,
    _compute_ev,
    _mode_value,
    _remove_vig,
    recommend_best_bets,
)
from line_tracker.models import BettingLine, BetType

# ---------------------------------------------------------------------------
# Helpers to build mock BettingLine objects
# ---------------------------------------------------------------------------

def _ml_line(sportsbook: str, home_odds: float, away_odds: float) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="basketball_nba",
        event="Lakers @ Celtics",
        bet_type=BetType.MONEYLINE,
        home_team="Celtics",
        away_team="Lakers",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=datetime(2025, 1, 1, 12, 0),
    )


def _spread_line(
    sportsbook: str,
    home_spread: float,
    away_spread: float,
    home_price: float,
    away_price: float,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="basketball_nba",
        event="Lakers @ Celtics",
        bet_type=BetType.SPREAD,
        home_team="Celtics",
        away_team="Lakers",
        home_value=home_spread,
        away_value=away_spread,
        timestamp=datetime(2025, 1, 1, 12, 0),
        home_price=home_price,
        away_price=away_price,
    )


def _total_line(
    sportsbook: str,
    total: float,
    over_price: float,
    under_price: float,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="basketball_nba",
        event="Lakers @ Celtics",
        bet_type=BetType.TOTAL,
        home_team="Celtics",
        away_team="Lakers",
        home_value=total,
        away_value=total,
        timestamp=datetime(2025, 1, 1, 12, 0),
        home_price=over_price,
        away_price=under_price,
    )


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestRemoveVig:
    def test_balanced_juice(self):
        # -110 / -110 should give ~50/50 after vig removal
        pa, pb = _remove_vig(-110, -110)
        assert abs(pa - 0.5) < 0.001
        assert abs(pb - 0.5) < 0.001

    def test_heavy_favourite(self):
        # -300 implied = 0.75, +250 implied ≈ 0.2857
        # sum ≈ 1.0357 → pa_no_vig ≈ 0.7241
        pa, pb = _remove_vig(-300, 250)
        assert 0.72 < pa < 0.73
        assert 0.27 < pb < 0.28

    def test_sums_to_one(self):
        pa, pb = _remove_vig(150, -150)
        assert abs(pa + pb - 1.0) < 0.0001

    def test_even_odds(self):
        pa, pb = _remove_vig(100, 100)
        assert abs(pa - 0.5) < 0.001
        assert abs(pb - 0.5) < 0.001


class TestComputeEv:
    def test_positive_ev(self):
        # consensus prob 0.55, odds +100 (decimal 2.0)
        # EV = 0.55 * 1.0 - 0.45 * 1.0 = 0.10
        ev = _compute_ev(0.55, 100)
        assert abs(ev - 0.10) < 0.001

    def test_negative_ev(self):
        # consensus prob 0.45, odds -110 (decimal ≈ 1.909)
        ev = _compute_ev(0.45, -110)
        assert ev < 0

    def test_break_even(self):
        # consensus prob 0.5, odds +100 (decimal 2.0)
        ev = _compute_ev(0.5, 100)
        assert abs(ev) < 0.001


class TestModeValue:
    def test_simple_mode(self):
        assert _mode_value([3.5, 3.5, 4.0]) == 3.5

    def test_all_same(self):
        assert _mode_value([7.0, 7.0, 7.0]) == 7.0

    def test_tie_returns_one_of_them(self):
        result = _mode_value([3.5, 4.0])
        assert result in (3.5, 4.0)


# ---------------------------------------------------------------------------
# Moneyline recommendations
# ---------------------------------------------------------------------------


class TestMoneylineRecommendations:
    def test_basic_moneyline(self):
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -140, 120),
            _ml_line("BetMGM", -160, 140),
        ]
        recs = recommend_best_bets(lines, top_n=10)

        assert len(recs) == 2  # home + away
        assert all(isinstance(r, BetRecommendation) for r in recs)
        assert all(r.market == "moneyline" for r in recs)

        # Sorted by EV descending
        assert recs[0].ev >= recs[1].ev

    def test_best_sportsbook_identified(self):
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -140, 120),
            _ml_line("BetMGM", -160, 140),
        ]
        recs = recommend_best_bets(lines, top_n=10)

        home_rec = next(r for r in recs if r.side == "home")
        assert home_rec.best_sportsbook == "FanDuel"  # -140 best for home fav
        assert home_rec.best_odds == -140

        away_rec = next(r for r in recs if r.side == "away")
        assert away_rec.best_sportsbook == "BetMGM"  # +140 best for away dog
        assert away_rec.best_odds == 140

    def test_consensus_probs_sum_to_one(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -150, 130),
            _ml_line("C", -150, 130),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")
        away_rec = next(r for r in recs if r.side == "away")
        assert abs(home_rec.consensus_prob + away_rec.consensus_prob - 1.0) < 0.001

    def test_selection_names(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")
        away_rec = next(r for r in recs if r.side == "away")
        assert home_rec.selection == "Celtics"
        assert away_rec.selection == "Lakers"


# ---------------------------------------------------------------------------
# Spread recommendations
# ---------------------------------------------------------------------------


class TestSpreadRecommendations:
    def test_basic_spread(self):
        lines = [
            _spread_line("DraftKings", -3.5, 3.5, -110, -110),
            _spread_line("FanDuel", -3.5, 3.5, -105, -115),
            _spread_line("BetMGM", -3.5, 3.5, -108, -112),
        ]
        recs = recommend_best_bets(lines, top_n=10)

        spread_recs = [r for r in recs if r.market == "spread"]
        assert len(spread_recs) == 2

        home_spread = next(r for r in spread_recs if r.side == "home")
        assert home_spread.line == -3.5
        assert home_spread.best_sportsbook == "FanDuel"  # -105 best juice
        assert home_spread.best_odds == -105

    def test_mixed_lines_uses_mode(self):
        """When books have different spread lines, use the most common."""
        lines = [
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -110, -110),
            _spread_line("C", -3.0, 3.0, -115, -105),  # different line
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        # Mode is -3.5 (2 books vs 1), away_spread is complementary 3.5
        assert all(r.line in (-3.5, 3.5) for r in spread_recs)

    def test_no_prices_returns_empty(self):
        lines = [
            BettingLine(
                sportsbook="A", sport="nba", event="X @ Y",
                bet_type=BetType.SPREAD, home_team="Y", away_team="X",
                home_value=-3.5, away_value=3.5,
                timestamp=datetime(2025, 1, 1),
                home_price=None, away_price=None,
            ),
            BettingLine(
                sportsbook="B", sport="nba", event="X @ Y",
                bet_type=BetType.SPREAD, home_team="Y", away_team="X",
                home_value=-3.5, away_value=3.5,
                timestamp=datetime(2025, 1, 1),
                home_price=None, away_price=None,
            ),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        assert len(recs) == 0


# ---------------------------------------------------------------------------
# Total recommendations
# ---------------------------------------------------------------------------


class TestTotalRecommendations:
    def test_basic_total(self):
        lines = [
            _total_line("DraftKings", 220.5, -110, -110),
            _total_line("FanDuel", 220.5, -105, -115),
            _total_line("BetMGM", 220.5, -108, -112),
        ]
        recs = recommend_best_bets(lines, top_n=10)

        total_recs = [r for r in recs if r.market == "total"]
        assert len(total_recs) == 2

        over_rec = next(r for r in total_recs if r.side == "over")
        assert over_rec.line == 220.5
        assert over_rec.selection == "Over"
        assert over_rec.best_sportsbook == "FanDuel"  # -105 best

        under_rec = next(r for r in total_recs if r.side == "under")
        assert under_rec.selection == "Under"

    def test_mixed_totals_uses_mode(self):
        lines = [
            _total_line("A", 220.5, -110, -110),
            _total_line("B", 220.5, -112, -108),
            _total_line("C", 221.0, -105, -115),  # different total
        ]
        recs = recommend_best_bets(lines, top_n=10)
        total_recs = [r for r in recs if r.market == "total"]
        assert all(r.line == 220.5 for r in total_recs)


# ---------------------------------------------------------------------------
# Combined / top-level behaviour
# ---------------------------------------------------------------------------


class TestRecommendBestBets:
    def test_all_markets_ranked_top3(self):
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -140, 120),
            _spread_line("DraftKings", -3.5, 3.5, -110, -110),
            _spread_line("FanDuel", -3.5, 3.5, -105, -115),
            _total_line("DraftKings", 220.5, -110, -110),
            _total_line("FanDuel", 220.5, -105, -115),
        ]
        recs = recommend_best_bets(lines, top_n=3)
        assert len(recs) == 3
        assert recs[0].ev >= recs[1].ev >= recs[2].ev

    def test_empty_input(self):
        assert recommend_best_bets([]) == []

    def test_single_book_returns_empty(self):
        lines = [_ml_line("DraftKings", -150, 130)]
        assert recommend_best_bets(lines) == []

    def test_edge_pct_matches_ev(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert abs(r.edge_pct - r.ev * 100) < 0.01

    def test_line_is_none_for_moneyline(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.line is None

    def test_default_top_n_is_three(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -105, -115),
            _total_line("A", 220.5, -110, -110),
            _total_line("B", 220.5, -105, -115),
        ]
        recs = recommend_best_bets(lines)
        assert len(recs) == 3

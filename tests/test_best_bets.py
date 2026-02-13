"""Tests for best_bets module."""

from datetime import datetime

from line_tracker.best_bets import (
    BOOK_WEIGHTS,
    BetRecommendation,
    _compute_ev,
    _confidence_label,
    _mode_value,
    _remove_vig,
    _weight_for_book,
    _weighted_median,
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


class TestConfidenceLabel:
    def test_high_confidence_tight_probs(self):
        # Books agree closely — IQR ≈ 0.003 (<= 0.03)
        probs = [0.550, 0.552, 0.548, 0.551]
        assert _confidence_label(probs) == "High"

    def test_medium_confidence(self):
        # Moderate disagreement — IQR ≈ 0.05 (0.03 < IQR <= 0.06)
        probs = [0.50, 0.53, 0.47, 0.52]
        assert _confidence_label(probs) == "Medium"

    def test_low_confidence_wide_probs(self):
        # Wide disagreement — IQR ≈ 0.175 (> 0.06)
        probs = [0.40, 0.55, 0.60, 0.45]
        assert _confidence_label(probs) == "Low"

    def test_single_book_returns_low(self):
        assert _confidence_label([0.55]) == "Low"

    def test_identical_probs_returns_high(self):
        assert _confidence_label([0.50, 0.50, 0.50]) == "High"

    def test_two_books_close(self):
        # IQR of [0.55, 0.56] = 0.005 → High
        assert _confidence_label([0.55, 0.56]) == "High"

    def test_two_books_far_apart(self):
        # IQR of [0.40, 0.60] = 0.10 → Low
        assert _confidence_label([0.40, 0.60]) == "Low"

    def test_iqr_at_high_boundary(self):
        # [0.49, 0.50, 0.51, 0.52] → q1=0.4925, q3=0.5175, IQR=0.025 → High
        assert _confidence_label([0.49, 0.50, 0.51, 0.52]) == "High"

    def test_iqr_at_medium_boundary(self):
        # [0.48, 0.50, 0.52, 0.54] → q1=0.485, q3=0.535, IQR=0.05 → Medium
        assert _confidence_label([0.48, 0.50, 0.52, 0.54]) == "Medium"

    def test_iqr_just_above_medium(self):
        # [0.40, 0.50, 0.55, 0.65] → q1=0.425, q3=0.625, IQR=0.20 → Low
        assert _confidence_label([0.40, 0.50, 0.55, 0.65]) == "Low"

    def test_five_books_high(self):
        # Five books with tight agreement
        probs = [0.550, 0.551, 0.552, 0.553, 0.554]
        assert _confidence_label(probs) == "High"

    def test_five_books_low(self):
        # Five books with wide disagreement
        probs = [0.40, 0.45, 0.55, 0.60, 0.65]
        assert _confidence_label(probs) == "Low"


class TestConfidenceIntegration:
    def test_tight_moneylines_high_confidence(self):
        """All books post nearly identical odds → High confidence."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -150, 130),
            _ml_line("C", -150, 130),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.confidence == "High"

    def test_wide_moneylines_lower_confidence(self):
        """Books disagree significantly → Medium or Low confidence."""
        lines = [
            _ml_line("A", -200, 180),
            _ml_line("B", -120, 100),
            _ml_line("C", -300, 260),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.confidence in ("Medium", "Low")

    def test_confidence_field_always_present(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -105, -115),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.confidence in ("High", "Medium", "Low")


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

    def test_edge_pct_is_consensus_minus_breakeven(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            expected_edge = (r.consensus_prob - r.breakeven_prob) * 100
            assert abs(r.edge_pct - expected_edge) < 0.01

    def test_ev_per_100_matches_ev(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert abs(r.ev_per_100 - r.ev * 100) < 0.01

    def test_breakeven_prob_populated(self):
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert 0 < r.breakeven_prob < 1

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


# ---------------------------------------------------------------------------
# Weighted median helper
# ---------------------------------------------------------------------------


class TestWeightedMedian:
    def test_equal_weights_matches_median(self):
        """With equal weights, weighted median should equal plain median."""
        vals = [0.50, 0.55, 0.60]
        wts = [1.0, 1.0, 1.0]
        assert abs(_weighted_median(vals, wts) - 0.55) < 0.001

    def test_single_value(self):
        assert _weighted_median([0.42], [2.0]) == 0.42

    def test_heavy_weight_shifts_toward_that_value(self):
        """If one value has much higher weight it should become the median."""
        vals = [0.40, 0.60]
        wts = [10.0, 1.0]
        result = _weighted_median(vals, wts)
        # Heavily weighted toward 0.40
        assert result == 0.40

    def test_heavy_weight_other_direction(self):
        vals = [0.40, 0.60]
        wts = [1.0, 10.0]
        result = _weighted_median(vals, wts)
        assert result == 0.60

    def test_three_values_weighted(self):
        """Middle value with low weight — median shifts to heavy side."""
        vals = [0.45, 0.50, 0.55]
        wts = [5.0, 0.1, 1.0]
        result = _weighted_median(vals, wts)
        # Heavy weight on 0.45 → cumulative passes 50% at 0.45
        assert result == 0.45

    def test_exact_midpoint_averages(self):
        """When cumulative weight hits exactly 50%, average current and next."""
        vals = [0.40, 0.60]
        wts = [1.0, 1.0]
        result = _weighted_median(vals, wts)
        assert abs(result - 0.50) < 0.001


# ---------------------------------------------------------------------------
# Book weights
# ---------------------------------------------------------------------------


class TestBookWeights:
    def test_known_book_returns_configured_weight(self):
        assert _weight_for_book("Pinnacle") == 3.0
        assert _weight_for_book("DraftKings") == 1.0

    def test_unknown_book_returns_default(self):
        assert _weight_for_book("SomeObscureBook") == 1.0

    def test_book_weights_dict_is_not_empty(self):
        assert len(BOOK_WEIGHTS) > 0


# ---------------------------------------------------------------------------
# Weighted consensus integration — shifts toward higher-weight book
# ---------------------------------------------------------------------------


class TestWeightedConsensusIntegration:
    def test_weighted_consensus_shifts_toward_sharp_book(self):
        """Pinnacle (weight 3.0) should pull consensus toward its probability.

        Pinnacle posts -200/+180 → de-vigged home ≈ 0.6410
        DraftKings posts -140/+120 → de-vigged home ≈ 0.5588

        Plain median of those two = ~0.60.
        Weighted median should equal the Pinnacle value since it has 3x weight.
        """
        lines = [
            _ml_line("Pinnacle", -200, 180),
            _ml_line("DraftKings", -140, 120),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")

        # Pinnacle's de-vigged home prob
        pinnacle_prob = home_rec.unweighted_consensus_prob  # plain median of 2
        # Weighted consensus should be closer to Pinnacle's value
        # because Pinnacle has weight 3.0 vs DraftKings 1.0
        # With 2 values and weights 3:1, weighted median = Pinnacle's value
        from line_tracker.best_bets import _remove_vig

        pin_h, _ = _remove_vig(-200, 180)
        dk_h, _ = _remove_vig(-140, 120)

        # Weighted median with weights [3.0, 1.0] → should be Pinnacle's value
        assert abs(home_rec.consensus_prob - pin_h) < 0.001

    def test_unweighted_consensus_is_plain_median(self):
        """unweighted_consensus_prob should be the simple median."""
        lines = [
            _ml_line("Pinnacle", -200, 180),
            _ml_line("DraftKings", -140, 120),
            _ml_line("FanDuel", -170, 150),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")

        from statistics import median as std_median

        from line_tracker.best_bets import _remove_vig

        probs = []
        for ln in lines:
            ph, _ = _remove_vig(ln.home_value, ln.away_value)
            probs.append(ph)
        expected = std_median(probs)
        assert abs(home_rec.unweighted_consensus_prob - round(expected, 4)) < 0.001

    def test_weighted_differs_from_unweighted(self):
        """When books have unequal weights, the two consensus values should differ."""
        lines = [
            _ml_line("Pinnacle", -200, 180),   # weight 3.0
            _ml_line("DraftKings", -140, 120),  # weight 1.0
            _ml_line("FanDuel", -140, 120),     # weight 1.0
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")
        # Unweighted median of 3 values — middle is DK or FD (both same)
        # Weighted median should be pulled toward Pinnacle
        assert home_rec.consensus_prob != home_rec.unweighted_consensus_prob

    def test_equal_weight_books_match_unweighted(self):
        """If all books have equal weight, weighted == unweighted."""
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -140, 120),
            _ml_line("Caesars", -145, 125),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert abs(r.consensus_prob - r.unweighted_consensus_prob) < 0.001

    def test_spread_weighted_consensus(self):
        """Weighted consensus also works for spread markets."""
        lines = [
            _spread_line("Pinnacle", -3.5, 3.5, -105, -115),  # weight 3.0
            _spread_line("DraftKings", -3.5, 3.5, -115, -105),  # weight 1.0
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        assert len(spread_recs) == 2

        home_spread = next(r for r in spread_recs if r.side == "home")
        # Pinnacle has -105 juice → higher de-vigged home prob
        # With weight 3:1, consensus should match Pinnacle's prob
        from line_tracker.best_bets import _remove_vig

        pin_h, _ = _remove_vig(-105, -115)
        assert abs(home_spread.consensus_prob - pin_h) < 0.001

    def test_total_weighted_consensus(self):
        """Weighted consensus also works for total markets."""
        lines = [
            _total_line("Pinnacle", 220.5, -105, -115),  # weight 3.0
            _total_line("DraftKings", 220.5, -115, -105),  # weight 1.0
        ]
        recs = recommend_best_bets(lines, top_n=10)
        total_recs = [r for r in recs if r.market == "total"]
        assert len(total_recs) == 2

        over_rec = next(r for r in total_recs if r.side == "over")
        from line_tracker.best_bets import _remove_vig

        pin_over, _ = _remove_vig(-105, -115)
        assert abs(over_rec.consensus_prob - pin_over) < 0.001

"""Tests for best_bets module."""

from datetime import datetime

from line_tracker.best_bets import (
    BOOK_WEIGHTS,
    RECENCY_HALF_LIFE_MIN,
    BetRecommendation,
    _agreement_score,
    _best_line_group,
    _cap_weights,
    _compute_ev,
    _confidence_label,
    _coverage_score,
    _edge_score,
    _filter_outliers,
    _freshness_score,
    _line_weight,
    _mode_value,
    _quality_score,
    _quality_tier,
    _recency_multiplier,
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
        """When books have unequal weights, the two consensus values should differ.

        With weight capping at 40%, we need 4+ books with varied probabilities
        so that the cap doesn't collapse weighted to unweighted.
        """
        lines = [
            _ml_line("Pinnacle", -160, 140),   # weight 3.0, home ≈ 0.596
            _ml_line("DraftKings", -145, 125),  # weight 1.0, home ≈ 0.571
            _ml_line("FanDuel", -140, 120),     # weight 1.0, home ≈ 0.562
            _ml_line("Caesars", -135, 115),      # weight 1.0, home ≈ 0.554
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")
        # Unweighted median of 4 values = average of middle two
        # Weighted median should be pulled toward Pinnacle's higher prob
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


# ---------------------------------------------------------------------------
# Recency multiplier
# ---------------------------------------------------------------------------


class TestRecencyMultiplier:
    def test_fresh_line_returns_one(self):
        """A line fetched right now should have recency ≈ 1.0."""
        now = datetime(2025, 6, 1, 12, 0)
        assert abs(_recency_multiplier(now, now) - 1.0) < 0.001

    def test_one_half_life_old(self):
        """After exactly RECENCY_HALF_LIFE_MIN, recency ≈ exp(-1) ≈ 0.368."""
        from math import exp

        now = datetime(2025, 6, 1, 13, 0)  # 60 min later
        ts = datetime(2025, 6, 1, 12, 0)
        expected = exp(-1.0)
        assert abs(_recency_multiplier(ts, now) - expected) < 0.001

    def test_very_old_line_hits_floor(self):
        """Extremely old timestamps should clamp to the floor (0.01)."""
        now = datetime(2025, 6, 1, 12, 0)
        ts = datetime(2020, 1, 1, 0, 0)  # years ago
        assert _recency_multiplier(ts, now) == 0.01

    def test_future_timestamp_returns_one(self):
        """If timestamp is somehow in the future, treat as fresh."""
        now = datetime(2025, 6, 1, 12, 0)
        future = datetime(2025, 6, 1, 13, 0)
        assert abs(_recency_multiplier(future, now) - 1.0) < 0.001

    def test_monotonically_decreasing(self):
        """Older lines should have lower recency than newer lines."""
        now = datetime(2025, 6, 1, 12, 0)
        fresh = _recency_multiplier(datetime(2025, 6, 1, 11, 50), now)  # 10m
        stale = _recency_multiplier(datetime(2025, 6, 1, 10, 0), now)  # 120m
        assert fresh > stale


# ---------------------------------------------------------------------------
# Combined line weight
# ---------------------------------------------------------------------------


class TestLineWeight:
    def test_fresh_sharp_book_highest(self):
        """Pinnacle fetched now should have highest weight."""
        now = datetime(2025, 6, 1, 12, 0)
        w = _line_weight("Pinnacle", now, now)
        assert abs(w - 3.0) < 0.01  # 3.0 * 1.0

    def test_stale_sharp_book_reduced(self):
        """Pinnacle fetched 60 min ago gets decayed."""
        from math import exp

        now = datetime(2025, 6, 1, 13, 0)
        ts = datetime(2025, 6, 1, 12, 0)
        expected = 3.0 * exp(-1.0)
        assert abs(_line_weight("Pinnacle", ts, now) - expected) < 0.01

    def test_stale_vs_fresh_same_book(self):
        """Fresh line from same book should outweigh stale line."""
        now = datetime(2025, 6, 1, 12, 0)
        fresh = _line_weight("DraftKings", datetime(2025, 6, 1, 11, 55), now)
        stale = _line_weight("DraftKings", datetime(2025, 6, 1, 10, 0), now)
        assert fresh > stale


# ---------------------------------------------------------------------------
# Recency-weighted consensus integration
# ---------------------------------------------------------------------------

def _ml_line_ts(
    sportsbook: str, home_odds: float, away_odds: float, ts: datetime,
) -> BettingLine:
    """Moneyline line helper with configurable timestamp."""
    return BettingLine(
        sportsbook=sportsbook,
        sport="basketball_nba",
        event="Lakers @ Celtics",
        bet_type=BetType.MONEYLINE,
        home_team="Celtics",
        away_team="Lakers",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=ts,
    )


class TestRecencyConsensusIntegration:
    def test_stale_outlier_influences_less(self):
        """A stale book with an outlier line should be down-weighted.

        Setup: two fresh books agree at -150/+130, one stale book posts
        an outlier at -200/+180.  Without recency weighting, the outlier
        would pull consensus toward its value.  With recency, it should
        contribute much less.
        """
        now = datetime(2025, 6, 1, 12, 0)
        fresh_ts = datetime(2025, 6, 1, 11, 55)  # 5 min ago
        stale_ts = datetime(2025, 6, 1, 8, 0)  # 4 hours ago

        lines = [
            _ml_line_ts("DraftKings", -150, 130, fresh_ts),
            _ml_line_ts("FanDuel", -150, 130, fresh_ts),
            _ml_line_ts("BetMGM", -200, 180, stale_ts),  # outlier, stale
        ]

        recs = recommend_best_bets(lines, top_n=10, now=now)
        home_rec = next(r for r in recs if r.side == "home")

        # Fresh books' de-vigged home prob
        fresh_h, _ = _remove_vig(-150, 130)
        # Stale outlier's de-vigged home prob
        stale_h, _ = _remove_vig(-200, 180)

        # Consensus should be much closer to the fresh books' value
        # because the stale outlier is down-weighted by recency decay
        dist_to_fresh = abs(home_rec.consensus_prob - fresh_h)
        dist_to_stale = abs(home_rec.consensus_prob - stale_h)
        assert dist_to_fresh < dist_to_stale

    def test_fresh_outlier_influences_more(self):
        """A fresh book with different odds pulls consensus when not capped.

        With weight capping (40% max per book), a single fresh book among
        many stale ones cannot fully dominate consensus. We use 5 books
        so the fresh book's influence is meaningful but constrained.
        """
        now = datetime(2025, 6, 1, 12, 0)
        stale_ts = datetime(2025, 6, 1, 8, 0)  # 4 hours ago
        fresh_ts = datetime(2025, 6, 1, 11, 58)  # 2 min ago

        lines = [
            _ml_line_ts("DraftKings", -150, 130, stale_ts),
            _ml_line_ts("FanDuel", -150, 130, stale_ts),
            _ml_line_ts("Caesars", -150, 130, stale_ts),
            _ml_line_ts("BetRivers", -155, 135, stale_ts),
            _ml_line_ts("BetMGM", -160, 140, fresh_ts),  # fresh, different odds
        ]

        recs = recommend_best_bets(lines, top_n=10, now=now)
        home_rec = next(r for r in recs if r.side == "home")

        stale_h, _ = _remove_vig(-150, 130)
        fresh_h, _ = _remove_vig(-160, 140)

        # Consensus should sit between stale and fresh values
        lo, hi = sorted([stale_h, fresh_h])
        assert lo - 0.001 <= home_rec.consensus_prob <= hi + 0.001

    def test_all_same_timestamp_preserves_book_weights(self):
        """When all lines are the same age, only book weights matter."""
        now = datetime(2025, 6, 1, 12, 0)
        ts = datetime(2025, 6, 1, 11, 50)  # all 10 min old

        lines = [
            _ml_line_ts("Pinnacle", -200, 180, ts),   # book weight 3.0
            _ml_line_ts("DraftKings", -140, 120, ts),  # book weight 1.0
        ]

        recs = recommend_best_bets(lines, top_n=10, now=now)
        home_rec = next(r for r in recs if r.side == "home")

        pin_h, _ = _remove_vig(-200, 180)
        # Equal recency → only book weight matters → Pinnacle dominates
        assert abs(home_rec.consensus_prob - pin_h) < 0.001

    def test_age_fields_populated(self):
        """newest_update_age_min and oldest_update_age_min should be set."""
        now = datetime(2025, 6, 1, 12, 0)
        ts1 = datetime(2025, 6, 1, 11, 50)  # 10 min ago
        ts2 = datetime(2025, 6, 1, 11, 30)  # 30 min ago

        lines = [
            _ml_line_ts("DraftKings", -150, 130, ts1),
            _ml_line_ts("FanDuel", -140, 120, ts2),
        ]

        recs = recommend_best_bets(lines, top_n=10, now=now)
        for r in recs:
            assert abs(r.newest_update_age_min - 10.0) < 0.1
            assert abs(r.oldest_update_age_min - 30.0) < 0.1

    def test_recency_weakens_stale_equal_weight_books(self):
        """Two equal-weight books, but one is stale — consensus shifts to fresh."""
        now = datetime(2025, 6, 1, 12, 0)
        fresh_ts = datetime(2025, 6, 1, 11, 58)
        stale_ts = datetime(2025, 6, 1, 8, 0)

        lines = [
            _ml_line_ts("DraftKings", -150, 130, fresh_ts),
            _ml_line_ts("FanDuel", -200, 180, stale_ts),
        ]

        recs = recommend_best_bets(lines, top_n=10, now=now)
        home_rec = next(r for r in recs if r.side == "home")

        dk_h, _ = _remove_vig(-150, 130)
        fd_h, _ = _remove_vig(-200, 180)

        # DK is fresh (weight ≈ 1.0), FD is 4h stale (weight ≈ 0.018)
        # Weighted median should pick DK's value
        assert abs(home_rec.consensus_prob - dk_h) < 0.001


# ---------------------------------------------------------------------------
# _best_line_group helper
# ---------------------------------------------------------------------------


class TestBestLineGroup:
    def test_clear_majority(self):
        """When one line has more books, pick it."""
        assert _best_line_group([-3.5, -3.5, -3.0]) == -3.5

    def test_all_same(self):
        assert _best_line_group([7.0, 7.0, 7.0]) == 7.0

    def test_tie_picks_closest_to_median(self):
        """Two lines with equal count — pick the one closest to median.

        Values: [-3.5, -3.0] → median = -3.25.
        Both are equidistant (0.25), so either is acceptable.
        """
        result = _best_line_group([-3.5, -3.0])
        assert result in (-3.5, -3.0)

    def test_tie_three_way_picks_closest_to_median(self):
        """Three distinct lines, each with 1 book.

        Values: [220.0, 220.5, 221.0] → median = 220.5.
        220.5 is the median itself, so it wins.
        """
        result = _best_line_group([220.0, 220.5, 221.0])
        assert result == 220.5

    def test_tie_two_groups_equal_count(self):
        """Two groups of 2 books each, pick line closest to median.

        Values: [-2.5, -2.5, -3.5, -3.5] → median = -3.0.
        |-2.5 - (-3.0)| = 0.5, |-3.5 - (-3.0)| = 0.5 → true tie.
        min() with key will pick the first (sorted order); either is valid.
        """
        result = _best_line_group([-2.5, -2.5, -3.5, -3.5])
        assert result in (-2.5, -3.5)

    def test_still_returns_mode_when_no_tie(self):
        """Backward compat: same behaviour as _mode_value when clear winner."""
        vals = [-3.5, -3.5, -3.5, -3.0, -4.0]
        assert _best_line_group(vals) == -3.5
        assert _mode_value(vals) == -3.5

    def test_near_tie_4v3_picks_median_close(self):
        """4 vs 3 split (within 1 book) should pick the median-closer group.

        4 books at -3.5, 3 books at -2.5.
        Median of all values = -3.5 (middle of 7 sorted values).
        -3.5 is closer to that median → -3.5 wins despite only 1 more book.
        """
        vals = [-3.5, -3.5, -3.5, -3.5, -2.5, -2.5, -2.5]
        assert _best_line_group(vals) == -3.5

    def test_near_tie_4v3_median_favours_smaller_group(self):
        """4 vs 3 split where the 3-book group wins via weighted median.

        4 retail books (weight 0.8) at -4.0, 3 sharp books (weight 3.0) at -3.0.
        Plain median = -4.0, but weighted median crosses 50% at -3.0.
        The near-tie tiebreak picks the group closest to weighted median → -3.0.
        """
        # Use weighted median: if the 3 sharp books (Pinnacle-like) are weighted
        # higher, the weighted median shifts toward -3.0
        vals = [-4.0, -4.0, -4.0, -4.0, -3.0, -3.0, -3.0]
        # Sharp books (weight 3.0 each) at -3.0 vs retail (weight 0.8 each) at -4.0
        weights = [0.8, 0.8, 0.8, 0.8, 3.0, 3.0, 3.0]
        # Weighted median: total_w = 3*3 + 4*0.8 = 12.2, half=6.1
        # Sorted by value: [(-4,0.8),(-4,0.8),(-4,0.8),(-4,0.8),(-3,3),(-3,3),(-3,3)]
        # Cumulative: 0.8, 1.6, 2.4, 3.2, 6.2 → crosses at -3.0
        result = _best_line_group(vals, weights)
        assert result == -3.0  # 3-book group wins because weighted median favours it

    def test_near_tie_without_weights_uses_plain_median(self):
        """Near-tie without weights should fall back to plain median."""
        # 4 at -3.5, 3 at -3.0
        vals = [-3.5, -3.5, -3.5, -3.5, -3.0, -3.0, -3.0]
        # plain median of 7 values sorted [-3.5,-3.5,-3.5,-3.5,-3,-3,-3] = -3.5
        assert _best_line_group(vals) == -3.5

    def test_clear_majority_by_2_no_near_tie(self):
        """5 vs 3 (diff of 2) is NOT a near-tie — the 5-book group wins outright."""
        vals = [-3.5, -3.5, -3.5, -3.5, -3.5, -3.0, -3.0, -3.0]
        assert _best_line_group(vals) == -3.5


# ---------------------------------------------------------------------------
# Line group selection integration — spread/total with split books
# ---------------------------------------------------------------------------


class TestLineGroupIntegration:
    def test_spread_tie_uses_median_tiebreak(self):
        """2 books at -3.5, 2 books at -3.0 → pick closer to median spread."""
        lines = [
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -110, -110),
            _spread_line("C", -3.0, 3.0, -110, -110),
            _spread_line("D", -3.0, 3.0, -110, -110),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        # median of [-3.5, -3.5, -3.0, -3.0] = -3.25
        # |-3.0 - (-3.25)| = 0.25, |-3.5 - (-3.25)| = 0.25
        # True tie — either is acceptable
        home_spread = next(r for r in spread_recs if r.side == "home")
        assert home_spread.line in (-3.5, -3.0)

    def test_spread_clear_majority_wins(self):
        """3 books at -3.5, 1 at -3.0 → -3.5 group wins."""
        lines = [
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -108, -112),
            _spread_line("C", -3.5, 3.5, -105, -115),
            _spread_line("D", -3.0, 3.0, -110, -110),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        home_spread = next(r for r in spread_recs if r.side == "home")
        assert home_spread.line == -3.5

    def test_total_tie_uses_median_tiebreak(self):
        """2 books at 220.5, 2 books at 221.0 → pick closest to median."""
        lines = [
            _total_line("A", 220.5, -110, -110),
            _total_line("B", 220.5, -110, -110),
            _total_line("C", 221.0, -110, -110),
            _total_line("D", 221.0, -110, -110),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        total_recs = [r for r in recs if r.market == "total"]
        over_rec = next(r for r in total_recs if r.side == "over")
        # median of [220.5, 220.5, 221.0, 221.0] = 220.75
        # |220.5 - 220.75| = 0.25, |221.0 - 220.75| = 0.25  → either ok
        assert over_rec.line in (220.5, 221.0)

    def test_books_used_count_and_total_books_count_spread(self):
        """Verify books_used_count and total_books_count on spread recs."""
        lines = [
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -105, -115),
            _spread_line("C", -3.0, 3.0, -110, -110),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        for r in spread_recs:
            assert r.books_used_count == 2  # A and B at -3.5
            assert r.total_books_count == 3

    def test_books_used_count_and_total_books_count_total(self):
        """Verify books_used_count and total_books_count on total recs."""
        lines = [
            _total_line("A", 220.5, -110, -110),
            _total_line("B", 220.5, -112, -108),
            _total_line("C", 221.0, -105, -115),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        total_recs = [r for r in recs if r.market == "total"]
        for r in total_recs:
            assert r.books_used_count == 2  # A and B at 220.5
            assert r.total_books_count == 3

    def test_moneyline_books_count_all_used(self):
        """Moneyline uses all books (no line grouping)."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
            _ml_line("C", -160, 140),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.books_used_count == 3
            assert r.total_books_count == 3

    def test_all_books_same_line_counts_match(self):
        """When all books agree on the line, used == total."""
        lines = [
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -105, -115),
            _spread_line("C", -3.5, 3.5, -108, -112),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        for r in spread_recs:
            assert r.books_used_count == 3
            assert r.total_books_count == 3

    def test_spread_near_tie_4v3_picks_median_close(self):
        """4 books at -3.5 vs 3 at -3.0 (near-tie).

        Tiebreak uses weighted median of all line values. With equal-weight
        books the plain median of 7 values [-3.5,-3.5,-3.5,-3.5,-3,-3,-3]
        = -3.5, so -3.5 is chosen.
        """
        lines = [
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -108, -112),
            _spread_line("C", -3.5, 3.5, -112, -108),
            _spread_line("D", -3.5, 3.5, -110, -110),
            _spread_line("E", -3.0, 3.0, -110, -110),
            _spread_line("F", -3.0, 3.0, -108, -112),
            _spread_line("G", -3.0, 3.0, -110, -110),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        home_spread = next(r for r in spread_recs if r.side == "home")
        assert home_spread.line == -3.5
        assert home_spread.books_used_count <= 4  # from the -3.5 group
        assert home_spread.total_books_count == 7

    def test_total_near_tie_4v3_picks_median_close(self):
        """4 books at 220.5 vs 3 at 221.0 (near-tie), median favours 220.5."""
        lines = [
            _total_line("A", 220.5, -110, -110),
            _total_line("B", 220.5, -108, -112),
            _total_line("C", 220.5, -112, -108),
            _total_line("D", 220.5, -110, -110),
            _total_line("E", 221.0, -110, -110),
            _total_line("F", 221.0, -108, -112),
            _total_line("G", 221.0, -110, -110),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        total_recs = [r for r in recs if r.market == "total"]
        over_rec = next(r for r in total_recs if r.side == "over")
        assert over_rec.line == 220.5
        assert over_rec.total_books_count == 7

    def test_spread_clear_majority_by_2_no_near_tie(self):
        """5 vs 3 (diff of 2) is not a near-tie — 5-book group wins outright."""
        lines = [
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -108, -112),
            _spread_line("C", -3.5, 3.5, -112, -108),
            _spread_line("D", -3.5, 3.5, -110, -110),
            _spread_line("E", -3.5, 3.5, -106, -114),
            _spread_line("F", -3.0, 3.0, -110, -110),
            _spread_line("G", -3.0, 3.0, -108, -112),
            _spread_line("H", -3.0, 3.0, -110, -110),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        home_spread = next(r for r in spread_recs if r.side == "home")
        assert home_spread.line == -3.5


# ---------------------------------------------------------------------------
# Quality score helpers — unit tests
# ---------------------------------------------------------------------------


class TestEdgeScore:
    def test_zero_edge(self):
        assert _edge_score(0.0) == 0.0

    def test_negative_edge(self):
        assert _edge_score(-1.0) == 0.0

    def test_half_percent(self):
        assert abs(_edge_score(0.5) - 35.0) < 0.01

    def test_one_percent(self):
        assert abs(_edge_score(1.0) - 55.0) < 0.01

    def test_two_percent(self):
        assert abs(_edge_score(2.0) - 75.0) < 0.01

    def test_three_percent(self):
        assert abs(_edge_score(3.0) - 85.0) < 0.01

    def test_five_percent(self):
        assert abs(_edge_score(5.0) - 95.0) < 0.01

    def test_above_seven_capped_at_100(self):
        assert _edge_score(10.0) == 100.0

    def test_interpolation_between_breakpoints(self):
        # 1.5% is midway between 1% (55) and 2% (75) → 65
        assert abs(_edge_score(1.5) - 65.0) < 0.01


class TestAgreementScore:
    def test_tight_many_books_fresh(self):
        # IQR <= 0.03 → base 90, std tiny → no penalty, books=5 < 6 → -10
        probs = [0.550, 0.552, 0.548, 0.551, 0.549]
        assert _agreement_score(probs, 5, stalest_age_min=10.0) == 80.0

    def test_tight_many_books_6_fresh(self):
        # IQR <= 0.03 → base 90, std tiny → no penalty, books=6 → no penalty
        probs = [0.550, 0.552, 0.548, 0.551, 0.549, 0.550]
        assert _agreement_score(probs, 6, stalest_age_min=10.0) == 90.0

    def test_tight_few_books(self):
        # IQR <= 0.03 → base 90, std tiny → no penalty, books=3 < 6 → -10
        probs = [0.550, 0.552, 0.548]
        assert _agreement_score(probs, 3, stalest_age_min=10.0) == 80.0

    def test_tight_stale(self):
        # IQR <= 0.03 → base 90, std tiny, books=5 < 6 → -10, stalest > 60 → -10
        probs = [0.550, 0.552, 0.548, 0.551, 0.549]
        assert _agreement_score(probs, 5, stalest_age_min=90.0) == 70.0

    def test_moderate_dispersion(self):
        # IQR 0.03–0.06 → base 70, std ≈ 0.026 → no penalty, books < 6 → -10
        probs = [0.48, 0.50, 0.52, 0.54]
        assert _agreement_score(probs, 4, stalest_age_min=10.0) == 60.0

    def test_wide_dispersion(self):
        # IQR > 0.06 → base 40, std ≈ 0.091 > 0.08 → -25, books < 6 → -10
        probs = [0.40, 0.55, 0.60, 0.45]
        assert _agreement_score(probs, 3, stalest_age_min=10.0) == 5.0

    def test_single_book(self):
        # < 2 probs → base 40, no std (len<2), books < 6 → -10
        assert _agreement_score([0.55], 1, stalest_age_min=10.0) == 30.0

    def test_clamp_floor(self):
        # Wide dispersion + std penalty + few books + stale → clamped at 0
        probs = [0.40, 0.55, 0.60, 0.45]
        assert _agreement_score(probs, 3, stalest_age_min=90.0) == 0.0

    def test_std_penalty_medium(self):
        # IQR tight but std between 0.05–0.08 triggers -15
        # Use values where IQR is small but tails push std > 0.05
        probs = [0.50, 0.50, 0.50, 0.50, 0.50, 0.38, 0.62]
        # q1=0.50, q3=0.50, IQR=0.0 → base 90
        # std ≈ 0.072 > 0.05 → -15
        # books=7 >= 6 → no penalty, fresh → no penalty
        assert _agreement_score(probs, 7, stalest_age_min=10.0) == 75.0

    def test_std_penalty_large(self):
        # std > 0.08 triggers -25
        probs = [0.50, 0.50, 0.50, 0.50, 0.50, 0.30, 0.70]
        # IQR=0.0 → base 90, std ≈ 0.112 > 0.08 → -25
        # books=7 >= 6 → no penalty
        assert _agreement_score(probs, 7, stalest_age_min=10.0) == 65.0

    def test_tight_beats_wide_with_similar_iqr(self):
        """Tight distribution scores higher than wide even when IQR is similar.

        Both sets have IQR near zero (all middle values identical), but the
        wide set has extreme tails that raise std and trigger a penalty.
        """
        # Tight: all values clustered, IQR≈0, std tiny
        tight = [0.50, 0.50, 0.51, 0.50, 0.49, 0.50, 0.50]
        # Wide: same IQR≈0 (middle 50% identical) but outlier tails → high std
        wide = [0.50, 0.50, 0.50, 0.50, 0.50, 0.35, 0.65]
        score_tight = _agreement_score(tight, 7, stalest_age_min=10.0)
        score_wide = _agreement_score(wide, 7, stalest_age_min=10.0)
        assert score_tight > score_wide


class TestCoverageScore:
    def test_full_coverage(self):
        assert _coverage_score(5, 5) == 100.0

    def test_half_coverage(self):
        assert abs(_coverage_score(3, 6) - 50.0) < 0.01

    def test_zero_total_safe(self):
        assert _coverage_score(0, 0) == 0.0

    def test_partial_coverage(self):
        assert abs(_coverage_score(2, 5) - 40.0) < 0.01

    def test_thin_market_4_of_4_capped_at_70(self):
        """4 total books with 4 used should NOT produce coverage_score 100."""
        score = _coverage_score(4, 4)
        assert score <= 70.0

    def test_thin_market_3_of_3_capped_at_70(self):
        """3/3 books should also be capped."""
        score = _coverage_score(3, 3)
        assert score <= 70.0

    def test_5_total_books_not_capped(self):
        """5+ total books should not be capped."""
        assert _coverage_score(5, 5) == 100.0
        assert _coverage_score(4, 5) == 80.0


class TestFreshnessScore:
    def test_very_fresh(self):
        assert _freshness_score(5.0, 8.0) == 95.0

    def test_moderate_age(self):
        assert _freshness_score(5.0, 25.0) == 75.0

    def test_hour_old(self):
        assert _freshness_score(10.0, 55.0) == 60.0

    def test_stale(self):
        assert _freshness_score(30.0, 90.0) == 40.0


class TestQualityScoreComposite:
    def test_perfect_scores(self):
        # All subscores at 100
        assert _quality_score(100, 100, 100, 100) == 100

    def test_zero_scores(self):
        assert _quality_score(0, 0, 0, 0) == 0

    def test_weights_sum_to_one(self):
        # 0.45 + 0.25 + 0.20 + 0.10 = 1.0
        # All subscores at 80 → quality = 80
        assert _quality_score(80, 80, 80, 80) == 80


class TestQualityTier:
    def test_elite(self):
        assert _quality_tier(85) == "Elite"
        assert _quality_tier(100) == "Elite"

    def test_strong(self):
        assert _quality_tier(70) == "Strong"
        assert _quality_tier(84) == "Strong"

    def test_moderate(self):
        assert _quality_tier(55) == "Moderate"
        assert _quality_tier(69) == "Moderate"

    def test_thin(self):
        assert _quality_tier(54) == "Thin"
        assert _quality_tier(0) == "Thin"


# ---------------------------------------------------------------------------
# Quality score integration — end-to-end via recommend_best_bets
# ---------------------------------------------------------------------------


class TestQualityScoreIntegration:
    def test_high_edge_many_books_low_dispersion_quality_ge_80(self):
        """High edge + many agreeing books + low dispersion → quality >= 80."""
        now = datetime(2025, 6, 1, 12, 0)
        fresh_ts = datetime(2025, 6, 1, 11, 55)  # 5 min ago

        # 5 books agree at -150/+130 (tight consensus home ≈ 0.58).
        # One generous book at -120/+100 provides a big edge for the home bet:
        # best home odds = -120 (breakeven ≈ 0.545) vs consensus ≈ 0.58 → ~3.4% edge.
        # 6 books, tight IQR, fresh data, full coverage → high quality.
        lines = [
            _ml_line_ts("Pinnacle", -150, 130, fresh_ts),
            _ml_line_ts("DraftKings", -150, 130, fresh_ts),
            _ml_line_ts("FanDuel", -150, 130, fresh_ts),
            _ml_line_ts("BetMGM", -150, 130, fresh_ts),
            _ml_line_ts("Caesars", -150, 130, fresh_ts),
            _ml_line_ts("BetOnline", -120, 100, fresh_ts),  # generous home odds
        ]
        recs = recommend_best_bets(lines, top_n=10, now=now)
        assert len(recs) >= 1

        # All recs should have quality fields populated
        for r in recs:
            assert r.quality_score >= 0
            assert r.quality_tier in ("Elite", "Strong", "Moderate", "Thin")
            assert 0 <= r.edge_score <= 100
            assert 0 <= r.agreement_score <= 100
            assert 0 <= r.coverage_score <= 100
            assert 0 <= r.freshness_score <= 100

        # The highest-EV rec (home side with ~3.4% edge) should score >= 80.
        best_rec = recs[0]
        assert best_rec.quality_score >= 80

    def test_low_edge_few_books_stale_quality_le_50(self):
        """Negative edge + few books + stale data → quality <= 50."""
        now = datetime(2025, 6, 1, 12, 0)
        stale_ts = datetime(2025, 6, 1, 10, 0)  # 2 hours ago

        # Two books with identical vigged odds → both sides have negative edge
        # (consensus < breakeven due to vig).  Only 2 books, stale timestamps.
        lines = [
            _ml_line_ts("DraftKings", -150, 130, stale_ts),
            _ml_line_ts("FanDuel", -150, 130, stale_ts),
        ]
        recs = recommend_best_bets(lines, top_n=10, now=now)
        assert len(recs) >= 1

        for r in recs:
            assert r.quality_score <= 50

    def test_quality_fields_always_present(self):
        """Every recommendation should have quality fields populated."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
            _spread_line("A", -3.5, 3.5, -110, -110),
            _spread_line("B", -3.5, 3.5, -105, -115),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert isinstance(r.quality_score, int)
            assert r.quality_tier in ("Elite", "Strong", "Moderate", "Thin")
            assert isinstance(r.edge_score, float)
            assert isinstance(r.agreement_score, float)
            assert isinstance(r.coverage_score, float)
            assert isinstance(r.freshness_score, float)

    def test_quality_tier_matches_score(self):
        """Quality tier should be consistent with quality score."""
        lines = [
            _ml_line("A", -150, 130),
            _ml_line("B", -140, 120),
            _ml_line("C", -160, 140),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            if r.market_unstable or r.books_used_count < 4:
                continue  # tier may be downgraded by caps
            if r.quality_score >= 85:
                assert r.quality_tier == "Elite"
            elif r.quality_score >= 70:
                assert r.quality_tier == "Strong"
            elif r.quality_score >= 55:
                assert r.quality_tier == "Moderate"
            else:
                assert r.quality_tier == "Thin"

    def test_thin_market_4_books_coverage_not_100(self):
        """4 total books with 4 used should NOT produce coverage_score 100."""
        now = datetime(2025, 6, 1, 12, 0)
        fresh_ts = datetime(2025, 6, 1, 11, 55)
        lines = [
            _ml_line_ts("DraftKings", -150, 130, fresh_ts),
            _ml_line_ts("FanDuel", -150, 130, fresh_ts),
            _ml_line_ts("BetMGM", -148, 128, fresh_ts),
            _ml_line_ts("Caesars", -152, 132, fresh_ts),
        ]
        recs = recommend_best_bets(lines, top_n=10, now=now)
        for r in recs:
            assert r.coverage_score <= 70.0, (
                f"coverage_score={r.coverage_score} with only 4 books"
            )

    def test_books_used_lt_4_prevents_strong_or_elite(self):
        """books_used < 4 should prevent Strong/Elite tier."""
        now = datetime(2025, 6, 1, 12, 0)
        fresh_ts = datetime(2025, 6, 1, 11, 55)
        # 3 books → all used (3 < 4) → tier capped at Moderate
        lines = [
            _ml_line_ts("DraftKings", -150, 130, fresh_ts),
            _ml_line_ts("FanDuel", -150, 130, fresh_ts),
            _ml_line_ts("BetMGM", -148, 128, fresh_ts),
        ]
        recs = recommend_best_bets(lines, top_n=10, now=now)
        for r in recs:
            assert r.quality_tier not in ("Elite", "Strong"), (
                f"tier={r.quality_tier} with only {r.books_used_count} books used"
            )
            assert r.quality_score <= 55, (
                f"quality_score={r.quality_score} with only {r.books_used_count} books used"
            )

    def test_books_used_ge_4_allows_strong(self):
        """With 5+ books used, Strong/Elite tiers should still be possible."""
        now = datetime(2025, 6, 1, 12, 0)
        fresh_ts = datetime(2025, 6, 1, 11, 55)
        # 6 books, tight consensus, generous edge → should reach Strong+
        lines = [
            _ml_line_ts("Pinnacle", -150, 130, fresh_ts),
            _ml_line_ts("DraftKings", -150, 130, fresh_ts),
            _ml_line_ts("FanDuel", -150, 130, fresh_ts),
            _ml_line_ts("BetMGM", -150, 130, fresh_ts),
            _ml_line_ts("Caesars", -150, 130, fresh_ts),
            _ml_line_ts("BetOnline", -120, 100, fresh_ts),  # generous edge
        ]
        recs = recommend_best_bets(lines, top_n=10, now=now)
        best = recs[0]
        assert best.quality_tier in ("Strong", "Elite"), (
            f"tier={best.quality_tier} should allow Strong/Elite with {best.books_used_count} books"
        )


# ---------------------------------------------------------------------------
# Outlier filtering — unit tests
# ---------------------------------------------------------------------------


class TestFilterOutliers:
    def test_no_outliers_keeps_all(self):
        """Books with similar probs should all be kept."""
        probs = [0.50, 0.51, 0.52, 0.49, 0.50]
        keep, unstable = _filter_outliers(probs)
        assert keep == [0, 1, 2, 3, 4]
        assert not unstable

    def test_single_outlier_removed(self):
        """One extreme value should be filtered out."""
        # median ≈ 0.50; 0.80 has rel_dev = |0.80-0.50|/0.50 = 0.60 > 0.15
        probs = [0.50, 0.51, 0.49, 0.50, 0.80]
        keep, unstable = _filter_outliers(probs)
        assert 4 not in keep  # the 0.80 outlier
        assert len(keep) == 4
        assert not unstable  # only 1 of 5 removed (20% < 30%), 4 books remain

    def test_prob_below_floor_filtered(self):
        """Probability below 0.01 should be filtered."""
        probs = [0.50, 0.51, 0.49, 0.50, 0.005]
        keep, unstable = _filter_outliers(probs)
        assert 4 not in keep

    def test_prob_above_ceiling_filtered(self):
        """Probability above 0.99 should be filtered."""
        probs = [0.50, 0.51, 0.49, 0.50, 0.995]
        keep, unstable = _filter_outliers(probs)
        assert 4 not in keep

    def test_market_unstable_when_too_many_filtered(self):
        """If >30% of books are filtered, market_unstable should be True."""
        # 3 of 5 are outliers → 60% filtered → unstable
        probs = [0.50, 0.51, 0.80, 0.85, 0.90]
        keep, unstable = _filter_outliers(probs)
        assert unstable

    def test_market_unstable_when_fewer_than_4_remain(self):
        """If fewer than 4 books remain after filtering, market_unstable."""
        # Only 3 books total, all kept → 3 < 4 → unstable
        probs = [0.50, 0.51, 0.52]
        keep, unstable = _filter_outliers(probs)
        assert len(keep) == 3
        assert unstable  # books_after < 4

    def test_single_value_returns_it(self):
        """Single value should be kept without error."""
        keep, unstable = _filter_outliers([0.55])
        assert keep == [0]
        assert not unstable

    def test_empty_list(self):
        """Empty list should return empty without error."""
        keep, unstable = _filter_outliers([])
        assert keep == []
        assert not unstable

    def test_all_identical_probs_no_filtering(self):
        """Identical probabilities → no outliers."""
        probs = [0.55, 0.55, 0.55, 0.55, 0.55]
        keep, unstable = _filter_outliers(probs)
        assert len(keep) == 5
        assert not unstable


class TestOutlierFilteringIntegration:
    def test_extreme_outlier_excluded_from_consensus(self):
        """One extreme outlier book should be removed and not dominate EV.

        Setup: 5 books agree around -150/+130 (home ≈ 0.58), one book posts
        an extreme -500/+400 (home ≈ 0.77). Without filtering the outlier
        would pull consensus higher. With filtering it should be excluded.
        """
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -150, 130),
            _ml_line("BetMGM", -148, 128),
            _ml_line("Caesars", -152, 132),
            _ml_line("Pinnacle", -150, 130),
            _ml_line("Outlier", -500, 400),  # extreme outlier
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")

        # The consensus should reflect the 5 agreeing books, not the outlier
        normal_prob, _ = _remove_vig(-150, 130)
        outlier_prob, _ = _remove_vig(-500, 400)

        # Consensus should be close to normal_prob, far from outlier_prob
        dist_to_normal = abs(home_rec.consensus_prob - normal_prob)
        dist_to_outlier = abs(home_rec.consensus_prob - outlier_prob)
        assert dist_to_normal < dist_to_outlier

        # Outlier flag should be set
        assert home_rec.outlier_filtered is True
        # 5 of 6 books kept → books_used_count == 5
        assert home_rec.books_used_count == 5

    def test_market_unstable_triggers_on_too_many_outliers(self):
        """If too many books are filtered, market_unstable should be True
        and the quality tier should be downgraded.
        """
        # 3 of 5 books are extreme outliers → >30% filtered → unstable
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -150, 130),
            _ml_line("OutlierA", -500, 400),
            _ml_line("OutlierB", -600, 500),
            _ml_line("OutlierC", -700, 600),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        home_rec = next(r for r in recs if r.side == "home")

        assert home_rec.market_unstable is True
        assert home_rec.outlier_filtered is True
        # Agreement score capped at 60 for unstable markets
        assert home_rec.agreement_score <= 60.0

    def test_no_outliers_flags_false(self):
        """When all books agree, no outlier flags should be set."""
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -150, 130),
            _ml_line("BetMGM", -150, 130),
            _ml_line("Caesars", -150, 130),
            _ml_line("Pinnacle", -150, 130),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            assert r.outlier_filtered is False
            assert r.market_unstable is False

    def test_outlier_not_used_for_ev(self):
        """The outlier book's probability should not inflate EV.

        An outlier posting very high home prob would inflate consensus_prob,
        making the home side look like a better bet than it is. After
        filtering, EV should be lower (closer to the true market).
        """
        normal_lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -150, 130),
            _ml_line("BetMGM", -148, 128),
            _ml_line("Caesars", -152, 132),
            _ml_line("Pinnacle", -150, 130),
        ]
        # Same lines + extreme outlier
        outlier_lines = normal_lines + [_ml_line("Outlier", -500, 400)]

        recs_normal = recommend_best_bets(normal_lines, top_n=10)
        recs_outlier = recommend_best_bets(outlier_lines, top_n=10)

        home_normal = next(r for r in recs_normal if r.side == "home")
        home_outlier = next(r for r in recs_outlier if r.side == "home")

        # With filtering, adding an outlier should NOT significantly change
        # the consensus probability (the outlier is removed)
        assert abs(home_normal.consensus_prob - home_outlier.consensus_prob) < 0.02

    def test_unstable_market_tier_downgraded(self):
        """Unstable markets should have their quality tier downgraded."""
        # Only 2 books remain after filtering → market_unstable
        lines = [
            _ml_line("DraftKings", -150, 130),
            _ml_line("FanDuel", -150, 130),
            _ml_line("OutlierA", -500, 400),
            _ml_line("OutlierB", -600, 500),
            _ml_line("OutlierC", -700, 600),
        ]
        recs = recommend_best_bets(lines, top_n=10)
        for r in recs:
            if r.market_unstable:
                # Tier should never be Elite when market is unstable
                assert r.quality_tier != "Elite"

    def test_spread_outlier_filtered(self):
        """Outlier filtering should also apply to spread markets."""
        lines = [
            _spread_line("DraftKings", -3.5, 3.5, -110, -110),
            _spread_line("FanDuel", -3.5, 3.5, -108, -112),
            _spread_line("BetMGM", -3.5, 3.5, -110, -110),
            _spread_line("Caesars", -3.5, 3.5, -110, -110),
            _spread_line("Pinnacle", -3.5, 3.5, -110, -110),
            _spread_line("Outlier", -3.5, 3.5, -300, 250),  # extreme juice
        ]
        recs = recommend_best_bets(lines, top_n=10)
        spread_recs = [r for r in recs if r.market == "spread"]
        assert len(spread_recs) == 2
        for r in spread_recs:
            assert r.outlier_filtered is True

    def test_total_outlier_filtered(self):
        """Outlier filtering should also apply to total markets."""
        lines = [
            _total_line("DraftKings", 220.5, -110, -110),
            _total_line("FanDuel", 220.5, -108, -112),
            _total_line("BetMGM", 220.5, -110, -110),
            _total_line("Caesars", 220.5, -110, -110),
            _total_line("Pinnacle", 220.5, -110, -110),
            _total_line("Outlier", 220.5, -300, 250),  # extreme juice
        ]
        recs = recommend_best_bets(lines, top_n=10)
        total_recs = [r for r in recs if r.market == "total"]
        assert len(total_recs) == 2
        for r in total_recs:
            assert r.outlier_filtered is True


# ---------------------------------------------------------------------------
# Weight capping — unit tests
# ---------------------------------------------------------------------------


class TestCapWeights:
    def test_equal_weights_unchanged(self):
        """Equal weights should stay equal after capping."""
        result = _cap_weights([1.0, 1.0, 1.0, 1.0])
        assert all(abs(w - 0.25) < 0.001 for w in result)

    def test_single_dominant_weight_capped(self):
        """A weight that would exceed 40% should be capped."""
        # Raw: [10.0, 1.0, 1.0] → normed [0.833, 0.083, 0.083]
        # After cap: [0.40, 0.083, 0.083] → renorm sums to ~0.567
        result = _cap_weights([10.0, 1.0, 1.0])
        assert result[0] <= 0.40 + 0.001
        assert abs(sum(result) - 1.0) < 0.001

    def test_no_weight_exceeds_cap(self):
        """No single weight should exceed the 40% cap."""
        result = _cap_weights([5.0, 3.0, 1.0, 0.5])
        for w in result:
            assert w <= 0.40 + 0.001

    def test_sums_to_one(self):
        """Capped weights should always sum to 1."""
        for raw in [[3.0, 1.0], [10.0, 1.0, 1.0], [5.0, 5.0, 1.0, 1.0, 1.0]]:
            result = _cap_weights(raw)
            assert abs(sum(result) - 1.0) < 0.001

    def test_empty_returns_empty(self):
        assert _cap_weights([]) == []

    def test_single_weight(self):
        """Single weight normalizes to 1.0 (cap skipped with < 3 books)."""
        result = _cap_weights([5.0])
        assert abs(result[0] - 1.0) < 0.001

    def test_two_books_cap_skipped(self):
        """With only 2 books, cap is skipped — just normalizes."""
        result = _cap_weights([3.0, 1.0])
        assert abs(result[0] - 0.75) < 0.001
        assert abs(result[1] - 0.25) < 0.001

    def test_two_equal_weights(self):
        """Two equal weights → 0.5 each (cap skipped with < 3 books)."""
        result = _cap_weights([1.0, 1.0])
        assert abs(result[0] - 0.5) < 0.001
        assert abs(result[1] - 0.5) < 0.001

    def test_all_zero_weights(self):
        """All-zero weights should distribute evenly."""
        result = _cap_weights([0.0, 0.0, 0.0])
        assert all(abs(w - 1.0 / 3) < 0.001 for w in result)

    def test_preserves_relative_order_of_small_weights(self):
        """Weights below the cap should maintain their relative ordering."""
        result = _cap_weights([10.0, 2.0, 1.0, 0.5])
        # The dominant weight is capped; the rest maintain relative order
        assert result[1] > result[2] > result[3]


class TestCapWeightsIntegration:
    def test_sharp_fresh_book_does_not_collapse_consensus(self):
        """One sharp fresh book among many stale books should not fully
        dominate consensus due to weight capping.

        Setup: Pinnacle (weight 3.0) is fresh (recency ~1.0, total ~3.0).
        Five other books are 4 hours stale (recency ~0.018, total ~0.018 each).
        Without capping, Pinnacle's share ≈ 3.0 / 3.09 ≈ 97% and consensus
        would collapse to Pinnacle's value.
        With capping at 40%, Pinnacle is limited and the 5 stale books
        collectively outweigh it, pulling consensus toward their value.
        """
        now = datetime(2025, 6, 1, 12, 0)
        fresh_ts = datetime(2025, 6, 1, 11, 58)  # 2 min ago
        stale_ts = datetime(2025, 6, 1, 8, 0)    # 4 hours ago

        lines = [
            _ml_line_ts("Pinnacle", -160, 140, fresh_ts),    # sharp + fresh
            _ml_line_ts("DraftKings", -140, 120, stale_ts),
            _ml_line_ts("FanDuel", -140, 120, stale_ts),
            _ml_line_ts("BetMGM", -140, 120, stale_ts),
            _ml_line_ts("Caesars", -140, 120, stale_ts),
            _ml_line_ts("BetRivers", -140, 120, stale_ts),
        ]

        recs = recommend_best_bets(lines, top_n=10, now=now)
        home_rec = next(r for r in recs if r.side == "home")

        pin_h, _ = _remove_vig(-160, 140)
        stale_h, _ = _remove_vig(-140, 120)

        # Consensus should NOT equal Pinnacle's value —
        # the capped weight prevents Pinnacle from dominating
        assert home_rec.consensus_prob != round(pin_h, 4)

        # With 5 stale books collectively having 60% weight, the weighted
        # median should be pulled toward the stale books' value
        dist_to_stale = abs(home_rec.consensus_prob - stale_h)
        dist_to_pin = abs(home_rec.consensus_prob - pin_h)
        assert dist_to_stale < dist_to_pin

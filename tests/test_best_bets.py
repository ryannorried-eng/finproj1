"""Tests for best_bets module."""

from datetime import datetime

from line_tracker.best_bets import (
    BOOK_WEIGHTS,
    RECENCY_HALF_LIFE_MIN,
    BetRecommendation,
    _agreement_score,
    _best_line_group,
    _compute_ev,
    _confidence_label,
    _coverage_score,
    _edge_score,
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
        """The opposite: a fresh outlier should pull consensus toward it."""
        now = datetime(2025, 6, 1, 12, 0)
        stale_ts = datetime(2025, 6, 1, 8, 0)  # 4 hours ago
        fresh_ts = datetime(2025, 6, 1, 11, 58)  # 2 min ago

        lines = [
            _ml_line_ts("DraftKings", -150, 130, stale_ts),
            _ml_line_ts("FanDuel", -150, 130, stale_ts),
            _ml_line_ts("BetMGM", -200, 180, fresh_ts),  # outlier, fresh
        ]

        recs = recommend_best_bets(lines, top_n=10, now=now)
        home_rec = next(r for r in recs if r.side == "home")

        stale_h, _ = _remove_vig(-150, 130)
        fresh_h, _ = _remove_vig(-200, 180)

        # Now the fresh outlier has high recency, stale books are down-weighted
        # Consensus should be closer to the fresh outlier
        dist_to_fresh = abs(home_rec.consensus_prob - fresh_h)
        dist_to_stale = abs(home_rec.consensus_prob - stale_h)
        assert dist_to_fresh < dist_to_stale

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
        # IQR <= 0.03 → base 90, 5+ books → no penalty, fresh → no penalty
        probs = [0.550, 0.552, 0.548, 0.551, 0.549]
        assert _agreement_score(probs, 5, stalest_age_min=10.0) == 90.0

    def test_tight_few_books(self):
        # IQR <= 0.03 → base 90, books < 5 → -10
        probs = [0.550, 0.552, 0.548]
        assert _agreement_score(probs, 3, stalest_age_min=10.0) == 80.0

    def test_tight_stale(self):
        # IQR <= 0.03 → base 90, 5+ books, stalest > 60 → -10
        probs = [0.550, 0.552, 0.548, 0.551, 0.549]
        assert _agreement_score(probs, 5, stalest_age_min=90.0) == 80.0

    def test_moderate_dispersion(self):
        # IQR 0.03–0.06 → base 70, books < 5 → -10
        probs = [0.48, 0.50, 0.52, 0.54]
        assert _agreement_score(probs, 4, stalest_age_min=10.0) == 60.0

    def test_wide_dispersion(self):
        # IQR > 0.06 → base 40, books < 5 → -10
        probs = [0.40, 0.55, 0.60, 0.45]
        assert _agreement_score(probs, 3, stalest_age_min=10.0) == 30.0

    def test_single_book(self):
        # < 2 probs → base 40, books < 5 → -10
        assert _agreement_score([0.55], 1, stalest_age_min=10.0) == 30.0

    def test_clamp_floor(self):
        # Wide dispersion + few books + stale → 40 - 10 - 10 = 20
        probs = [0.40, 0.55, 0.60, 0.45]
        assert _agreement_score(probs, 3, stalest_age_min=90.0) == 20.0


class TestCoverageScore:
    def test_full_coverage(self):
        assert _coverage_score(5, 5) == 100.0

    def test_half_coverage(self):
        assert abs(_coverage_score(3, 6) - 50.0) < 0.01

    def test_zero_total_safe(self):
        assert _coverage_score(0, 0) == 0.0

    def test_partial_coverage(self):
        assert abs(_coverage_score(2, 5) - 40.0) < 0.01


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
            if r.quality_score >= 85:
                assert r.quality_tier == "Elite"
            elif r.quality_score >= 70:
                assert r.quality_tier == "Strong"
            elif r.quality_score >= 55:
                assert r.quality_tier == "Moderate"
            else:
                assert r.quality_tier == "Thin"

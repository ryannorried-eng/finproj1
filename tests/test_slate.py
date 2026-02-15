"""Tests for the Daily Slate aggregator (slate module)."""

from datetime import datetime
from unittest.mock import patch

from line_tracker.best_bets import BetRecommendation
from line_tracker.models import BettingLine, BetType
from line_tracker.slate import (
    _EDGE_OUTLIER_THRESHOLD,
    _MIN_BOOKS,
    _TIER1_BASE_EDGE,
    _TIER1_SIGMA_MULT,
    _assign_tier,
    _passes_filters,
    _slate_score,
    build_daily_slate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 6, 1, 12, 0)


def _ml_line(
    sportsbook: str,
    home_odds: float,
    away_odds: float,
    event: str = "Lakers @ Celtics",
    commence_time: datetime | None = None,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="basketball_nba",
        event=event,
        bet_type=BetType.MONEYLINE,
        home_team="Celtics",
        away_team="Lakers",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=_NOW,
        commence_time=commence_time,
    )


def _make_rec(
    *,
    market: str = "moneyline",
    selection: str = "Celtics",
    edge_pct: float = 3.0,
    quality_score: int = 80,
    quality_tier: str = "Strong",
    confidence: str = "High",
    books_used_count: int = 5,
    newest_update_age_min: float = 10.0,
    oldest_update_age_min: float = 15.0,
    market_unstable: bool = False,
    market_volatility_sigma: float = 0.0,
    ev: float = 0.05,
    best_sportsbook: str = "FanDuel",
    best_odds: float = -110.0,
    line: float | None = None,
) -> BetRecommendation:
    return BetRecommendation(
        market=market,
        selection=selection,
        side="home",
        line=line,
        consensus_prob=0.55,
        best_sportsbook=best_sportsbook,
        best_odds=best_odds,
        breakeven_prob=0.52,
        ev=ev,
        edge_pct=edge_pct,
        ev_per_100=ev * 100,
        confidence=confidence,
        quality_score=quality_score,
        quality_tier=quality_tier,
        books_used_count=books_used_count,
        newest_update_age_min=newest_update_age_min,
        oldest_update_age_min=oldest_update_age_min,
        market_unstable=market_unstable,
        market_volatility_sigma=market_volatility_sigma,
    )


def _event_lines(event: str = "Lakers @ Celtics", commence: datetime | None = None):
    """Return a minimal list of lines for one event."""
    return [
        _ml_line("FanDuel", -150, 130, event=event, commence_time=commence),
        _ml_line("DraftKings", -145, 125, event=event, commence_time=commence),
    ]


# ---------------------------------------------------------------------------
# _slate_score
# ---------------------------------------------------------------------------


class TestSlateScore:
    def test_formula(self):
        # quality=80, edge=3.0 → 0.6*80 + 0.4*min(3,5)*20 = 48 + 24 = 72
        assert _slate_score(80, 3.0) == 72.0

    def test_edge_capped_at_5(self):
        # edge=10 → clamped to 5 → 0.6*50 + 0.4*5*20 = 30 + 40 = 70
        assert _slate_score(50, 10.0) == 70.0

    def test_zero_edge(self):
        # quality=60, edge=0 → 0.6*60 + 0 = 36
        assert _slate_score(60, 0.0) == 36.0

    def test_zero_quality(self):
        # quality=0, edge=5 → 0 + 0.4*5*20 = 40
        assert _slate_score(0, 5.0) == 40.0


# ---------------------------------------------------------------------------
# _assign_tier
# ---------------------------------------------------------------------------


class TestAssignTier:
    def test_tier1(self):
        tier, reasons = _assign_tier(
            80, 3.0, "High", False, rec_books_used=6,
            rec_quality_tier="Strong",
        )
        assert tier == "tier1"
        assert reasons == []

    def test_tier2_moderate_quality(self):
        tier, reasons = _assign_tier(55, 1.5, "Medium", False, rec_books_used=5)
        assert tier == "tier2"
        assert reasons == []

    def test_avoid_low_edge(self):
        tier, reasons = _assign_tier(80, 0.5, "High", False, rec_books_used=6)
        assert tier == "avoid"
        assert any("Edge too small" in r for r in reasons)

    def test_avoid_low_quality(self):
        tier, reasons = _assign_tier(20, 3.0, "High", False, rec_books_used=6)
        assert tier == "avoid"
        assert any("Low quality score" in r for r in reasons)

    def test_avoid_low_confidence(self):
        tier, reasons = _assign_tier(80, 3.0, "Low", False, rec_books_used=6)
        assert tier == "avoid"
        assert any("Low confidence (books disagree)" in r for r in reasons)

    def test_avoid_unstable_market(self):
        tier, reasons = _assign_tier(80, 3.0, "High", True, rec_books_used=6)
        assert tier == "avoid"
        assert any("Unstable market" in r for r in reasons)

    def test_avoid_multiple_reasons(self):
        tier, reasons = _assign_tier(
            20, 0.5, "Low", True, rec_books_used=2, rec_oldest_age_min=200.0,
        )
        assert tier == "avoid"
        # unstable, edge too small, low quality, low confidence,
        # too few books, stale lines
        assert len(reasons) == 6

    def test_tier2_boundary_quality(self):
        # quality exactly 40, edge exactly 1.0 → tier2
        tier, reasons = _assign_tier(40, 1.0, "Medium", False, rec_books_used=5)
        assert tier == "tier2"

    def test_tier1_boundary_edge_below_base(self):
        """Edge exactly 2.0 is below base 3.0 → tier2 even with Strong tier."""
        tier, reasons = _assign_tier(
            70, 2.0, "High", False, rec_books_used=6,
            rec_quality_tier="Strong",
        )
        assert tier == "tier2"

    def test_avoid_too_few_books(self):
        """Fewer than 4 books triggers avoid."""
        tier, reasons = _assign_tier(80, 3.0, "High", False, rec_books_used=3)
        assert tier == "avoid"
        assert any(f"Too few books (<{_MIN_BOOKS})" in r for r in reasons)

    def test_exactly_4_books_no_too_few_reason(self):
        """Exactly 4 books does NOT trigger the too-few-books reason."""
        tier, reasons = _assign_tier(
            80, 3.0, "High", False, rec_books_used=4,
            rec_quality_tier="Strong",
        )
        # No avoid reason for books, but quality_tier="Strong" + edge=3.0 → tier1
        assert not any("Too few books" in r for r in reasons)
        assert tier == "tier1"

    def test_avoid_stale_lines(self):
        """Oldest update > 120 min triggers avoid."""
        tier, reasons = _assign_tier(
            80, 3.0, "High", False, rec_books_used=6, rec_oldest_age_min=150.0,
        )
        assert tier == "avoid"
        assert any("Stale lines" in r for r in reasons)

    def test_stale_lines_boundary(self):
        """Exactly 120 min does NOT trigger stale-lines reason."""
        tier, reasons = _assign_tier(
            80, 3.0, "High", False, rec_books_used=6, rec_oldest_age_min=120.0,
            rec_quality_tier="Strong",
        )
        assert tier == "tier1"
        assert not any("Stale lines" in r for r in reasons)

    def test_avoid_edge_outlier_low_confidence(self):
        """High edge + low confidence triggers the outlier reason."""
        tier, reasons = _assign_tier(
            80, _EDGE_OUTLIER_THRESHOLD, "Low", False, rec_books_used=6,
        )
        assert tier == "avoid"
        assert any("Edge outlier" in r for r in reasons)
        # Also has the basic low-confidence reason
        assert any("Low confidence (books disagree)" in r for r in reasons)

    def test_no_edge_outlier_when_high_confidence(self):
        """High edge + high confidence does NOT trigger outlier reason."""
        tier, reasons = _assign_tier(
            80, 5.0, "High", False, rec_books_used=6,
            rec_quality_tier="Strong",
        )
        assert tier == "tier1"
        assert not any("Edge outlier" in r for r in reasons)

    def test_tier1_requires_high_confidence(self):
        """Tier1 requires confidence == 'High'."""
        tier, _ = _assign_tier(
            80, 4.0, "Medium", False, rec_books_used=6,
            rec_quality_tier="Strong",
        )
        assert tier == "tier2"

    def test_tier1_requires_elite_or_strong_tier(self):
        """Tier1 requires quality_tier in {Elite, Strong}."""
        tier, _ = _assign_tier(
            80, 4.0, "High", False, rec_books_used=6,
            rec_quality_tier="Moderate",
        )
        assert tier == "tier2"

    def test_tier1_with_elite_tier(self):
        """Elite quality_tier should also qualify for tier1."""
        tier, reasons = _assign_tier(
            90, 4.0, "High", False, rec_books_used=6,
            rec_quality_tier="Elite",
        )
        assert tier == "tier1"
        assert reasons == []


# ---------------------------------------------------------------------------
# _passes_filters
# ---------------------------------------------------------------------------


class TestPassesFilters:
    def _entry(self, **overrides):
        base = {
            "edge_pct": 3.0,
            "quality_score": 80,
            "market": "moneyline",
            "confidence": "High",
            "books_used": 5,
        }
        base.update(overrides)
        return base

    def test_no_filters(self):
        assert _passes_filters(self._entry(), {}) is True

    def test_min_edge_pass(self):
        assert _passes_filters(self._entry(edge_pct=3.0), {"min_edge": 2.0}) is True

    def test_min_edge_fail(self):
        assert _passes_filters(self._entry(edge_pct=1.0), {"min_edge": 2.0}) is False

    def test_min_quality_pass(self):
        assert _passes_filters(self._entry(quality_score=80), {"min_quality": 60}) is True

    def test_min_quality_fail(self):
        assert _passes_filters(self._entry(quality_score=30), {"min_quality": 60}) is False

    def test_markets_pass(self):
        assert _passes_filters(
            self._entry(market="spread"), {"markets": ["spread", "total"]}
        ) is True

    def test_markets_fail(self):
        assert _passes_filters(
            self._entry(market="moneyline"), {"markets": ["spread", "total"]}
        ) is False

    def test_hide_low_confidence_pass(self):
        assert _passes_filters(
            self._entry(confidence="High"), {"hide_low_confidence": True}
        ) is True

    def test_hide_low_confidence_fail(self):
        assert _passes_filters(
            self._entry(confidence="Low"), {"hide_low_confidence": True}
        ) is False

    def test_books_used_min_pass(self):
        assert _passes_filters(self._entry(books_used=5), {"books_used_min": 3}) is True

    def test_books_used_min_fail(self):
        assert _passes_filters(self._entry(books_used=2), {"books_used_min": 3}) is False

    def test_combined_filters(self):
        entry = self._entry(edge_pct=4.0, quality_score=90, market="spread")
        filters = {"min_edge": 2.0, "min_quality": 50, "markets": ["spread"]}
        assert _passes_filters(entry, filters) is True

    def test_combined_filters_one_fails(self):
        entry = self._entry(edge_pct=1.0, quality_score=90, market="spread")
        filters = {"min_edge": 2.0, "min_quality": 50, "markets": ["spread"]}
        assert _passes_filters(entry, filters) is False


# ---------------------------------------------------------------------------
# build_daily_slate – integration tests (mock recommend_best_bets)
# ---------------------------------------------------------------------------


class TestBuildDailySlate:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_basic_tier1(self, mock_rbb):
        """A high-quality recommendation lands in tier1."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong", confidence="High",
        )]
        lines = {"evt1": _event_lines()}
        result = build_daily_slate(lines)
        assert len(result["tier1"]) == 1
        assert result["tier1"][0]["event"] == "Lakers @ Celtics"
        assert result["tier1"][0]["tier"] == "tier1"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_basic_tier2(self, mock_rbb):
        """A moderate recommendation lands in tier2."""
        mock_rbb.return_value = [_make_rec(
            quality_score=55, edge_pct=1.5,
            quality_tier="Moderate", confidence="Medium",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["tier2"]) == 1
        assert result["tier2"][0]["tier"] == "tier2"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_basic_avoid(self, mock_rbb):
        """A low-quality recommendation lands in avoid."""
        mock_rbb.return_value = [_make_rec(quality_score=20, edge_pct=0.5)]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["avoid"]) == 1
        assert len(result["avoid"][0]["avoid_reasons"]) >= 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_metadata_attached(self, mock_rbb):
        """Slate entries carry event metadata."""
        ct = datetime(2025, 6, 2, 19, 0)
        mock_rbb.return_value = [_make_rec(
            books_used_count=6, newest_update_age_min=5.0,
            quality_tier="Strong", confidence="High",
        )]
        lines = {"evt1": _event_lines("Knicks @ Heat", commence=ct)}
        result = build_daily_slate(lines)
        entry = (result["tier1"] + result["tier2"] + result["avoid"])[0]
        assert entry["event"] == "Knicks @ Heat"
        assert entry["commence_time"] == ct
        assert entry["books_used"] == 6
        assert entry["updated_age_min"] == 5.0
        assert entry["quality_tier"] == "Strong"
        assert entry["market_volatility_sigma"] == 0.0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_slate_score_computed(self, mock_rbb):
        """slate_score matches the formula."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["tier1"][0]
        expected = 0.6 * 80 + 0.4 * min(3.0, 5) * 20
        assert entry["slate_score"] == expected

    @patch("line_tracker.slate.recommend_best_bets")
    def test_score_ordering(self, mock_rbb):
        """Entries are sorted by slate_score descending within each tier."""
        # Two tier1 recommendations with different scores
        def side_effect(lines):
            event_name = lines[0].event
            if "Knicks" in event_name:
                return [_make_rec(
                    quality_score=90, edge_pct=4.0,
                    quality_tier="Elite", confidence="High",
                )]
            return [_make_rec(
                quality_score=80, edge_pct=3.5,
                quality_tier="Strong", confidence="High",
            )]

        mock_rbb.side_effect = side_effect
        lines = {
            "evt1": _event_lines("Celtics @ Bucks"),
            "evt2": _event_lines("Knicks @ Heat"),
        }
        result = build_daily_slate(lines)
        tier1 = result["tier1"]
        assert len(tier1) == 2
        assert tier1[0]["slate_score"] >= tier1[1]["slate_score"]
        assert tier1[0]["event"] == "Knicks @ Heat"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_max_per_event_default(self, mock_rbb):
        """By default only the top 1 recommendation per event is taken."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=3.0, market="moneyline",
                      quality_tier="Strong", confidence="High"),
            _make_rec(quality_score=60, edge_pct=2.0, market="spread",
                      quality_tier="Moderate", confidence="Medium"),
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        total = len(result["tier1"]) + len(result["tier2"]) + len(result["avoid"])
        assert total == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_max_per_event_two(self, mock_rbb):
        """Setting max_per_event=2 takes 2 recommendations."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=3.0, market="moneyline",
                      quality_tier="Strong", confidence="High"),
            _make_rec(quality_score=60, edge_pct=2.0, market="spread",
                      quality_tier="Moderate", confidence="Medium"),
        ]
        result = build_daily_slate({"evt1": _event_lines()}, filters={"max_per_event": 2})
        total = len(result["tier1"]) + len(result["tier2"]) + len(result["avoid"])
        assert total == 2

    @patch("line_tracker.slate.recommend_best_bets")
    def test_filter_min_edge(self, mock_rbb):
        """Entries below min_edge are dropped entirely."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=1.0, quality_tier="Moderate",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"min_edge": 2.0}
        )
        total = len(result["tier1"]) + len(result["tier2"]) + len(result["avoid"])
        assert total == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_filter_markets(self, mock_rbb):
        """Only specified markets survive."""
        mock_rbb.return_value = [_make_rec(
            market="moneyline", quality_score=80, edge_pct=3.0,
            quality_tier="Strong", confidence="High",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"markets": ["spread", "total"]}
        )
        total = len(result["tier1"]) + len(result["tier2"]) + len(result["avoid"])
        assert total == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_filter_hide_low_confidence(self, mock_rbb):
        """Low confidence entries are hidden when requested."""
        mock_rbb.return_value = [_make_rec(
            confidence="Low", quality_score=80, edge_pct=3.0,
            quality_tier="Thin",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"hide_low_confidence": True}
        )
        total = len(result["tier1"]) + len(result["tier2"]) + len(result["avoid"])
        assert total == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_empty_input(self, mock_rbb):
        """Empty lines_by_event produces empty tiers."""
        result = build_daily_slate({})
        assert result == {"tier1": [], "tier2": [], "avoid": []}
        mock_rbb.assert_not_called()

    @patch("line_tracker.slate.recommend_best_bets")
    def test_no_recommendations(self, mock_rbb):
        """Event with no recommendations is skipped."""
        mock_rbb.return_value = []
        result = build_daily_slate({"evt1": _event_lines()})
        assert result == {"tier1": [], "tier2": [], "avoid": []}

    @patch("line_tracker.slate.recommend_best_bets")
    def test_multiple_events_mixed_tiers(self, mock_rbb):
        """Multiple events land in different tiers."""

        def side_effect(lines):
            event_name = lines[0].event
            if "Knicks" in event_name:
                return [_make_rec(
                    quality_score=90, edge_pct=4.0,
                    quality_tier="Elite", confidence="High",
                )]
            elif "Celtics" in event_name:
                return [_make_rec(
                    quality_score=50, edge_pct=1.5,
                    quality_tier="Moderate", confidence="Medium",
                )]
            return [_make_rec(
                quality_score=20, edge_pct=0.5,
                confidence="Low", quality_tier="Thin",
            )]

        mock_rbb.side_effect = side_effect
        lines = {
            "evt1": _event_lines("Knicks @ Heat"),
            "evt2": _event_lines("Celtics @ Bucks"),
            "evt3": _event_lines("Warriors @ Suns"),
        }
        result = build_daily_slate(lines)
        assert len(result["tier1"]) == 1
        assert len(result["tier2"]) == 1
        assert len(result["avoid"]) == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_reasons_populated(self, mock_rbb):
        """Avoid entries have descriptive reasons."""
        mock_rbb.return_value = [
            _make_rec(quality_score=20, edge_pct=0.5, confidence="Low",
                      market_unstable=True, quality_tier="Thin")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["avoid"][0]
        assert any("Unstable market" in r for r in entry["avoid_reasons"])
        assert any("Low confidence (books disagree)" in r for r in entry["avoid_reasons"])
        assert any("Edge too small" in r for r in entry["avoid_reasons"])
        assert any("Low quality score" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_stale_lines_integration(self, mock_rbb):
        """Stale lines cause avoid with descriptive reason."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=3.0, oldest_update_age_min=200.0,
                      quality_tier="Strong", confidence="High")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["avoid"][0]
        assert any("Stale lines" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_too_few_books_integration(self, mock_rbb):
        """Too few books causes avoid with descriptive reason."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=3.0, books_used_count=2,
                      quality_tier="Strong", confidence="High")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["avoid"][0]
        assert any("Too few books" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_edge_outlier_integration(self, mock_rbb):
        """High edge + low confidence triggers edge outlier reason."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=5.0, confidence="Low",
                      quality_tier="Thin")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["avoid"][0]
        assert any("Edge outlier" in r for r in entry["avoid_reasons"])
        assert any("Low confidence" in r for r in entry["avoid_reasons"])


# ---------------------------------------------------------------------------
# Dynamic edge floor tests
# ---------------------------------------------------------------------------


class TestDynamicEdgeFloor:
    def test_zero_sigma_uses_base_edge(self):
        """With sigma=0, dynamic floor equals the base edge (3.0%)."""
        # edge exactly at base → tier1
        tier, _ = _assign_tier(
            80, _TIER1_BASE_EDGE, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=0.0,
        )
        assert tier == "tier1"

    def test_below_base_edge_is_tier2(self):
        """Edge below 3.0% base with zero sigma → tier2."""
        tier, _ = _assign_tier(
            80, 2.99, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=0.0,
        )
        assert tier == "tier2"

    def test_sigma_raises_floor(self):
        """Non-zero sigma raises the dynamic floor above base edge.

        sigma=0.02 → floor = 3.0 + 1.2*0.02 = 3.024
        edge=3.01 < 3.024 → tier2
        """
        tier, _ = _assign_tier(
            80, 3.01, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=0.02,
        )
        assert tier == "tier2"

    def test_edge_above_raised_floor(self):
        """Edge above the sigma-raised floor → tier1.

        sigma=0.02 → floor = 3.0 + 1.2*0.02 = 3.024
        edge=3.03 >= 3.024 → tier1
        """
        tier, _ = _assign_tier(
            80, 3.03, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=0.02,
        )
        assert tier == "tier1"

    def test_high_sigma_needs_large_edge(self):
        """Large sigma pushes the floor much higher.

        sigma=1.0 → floor = 3.0 + 1.2*1.0 = 4.2
        edge=4.0 < 4.2 → tier2
        """
        tier, _ = _assign_tier(
            80, 4.0, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=1.0,
        )
        assert tier == "tier2"

    def test_high_sigma_edge_above(self):
        """Edge above the high-sigma floor → tier1.

        sigma=1.0 → floor = 4.2; edge=4.3 >= 4.2 → tier1
        """
        tier, _ = _assign_tier(
            80, 4.3, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=1.0,
        )
        assert tier == "tier1"

    def test_dynamic_floor_formula(self):
        """Verify the formula: floor = base + mult * sigma."""
        sigma = 0.05
        expected_floor = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma
        # edge just at the floor → tier1
        tier_at, _ = _assign_tier(
            80, expected_floor, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=sigma,
        )
        assert tier_at == "tier1"
        # edge just below the floor → tier2
        tier_below, _ = _assign_tier(
            80, expected_floor - 0.001, "High", False,
            rec_books_used=6, rec_quality_tier="Strong",
            rec_volatility_sigma=sigma,
        )
        assert tier_below == "tier2"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_dynamic_floor_integration(self, mock_rbb):
        """Integration: high sigma pushes a borderline rec from tier1 to tier2."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.1,
            quality_tier="Strong", confidence="High",
            market_volatility_sigma=0.1,  # floor = 3.0 + 1.2*0.1 = 3.12
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        # 3.1 < 3.12 → tier2
        assert len(result["tier2"]) == 1
        assert len(result["tier1"]) == 0

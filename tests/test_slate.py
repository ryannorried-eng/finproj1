"""Tests for the Daily Slate aggregator (slate module)."""

from datetime import datetime
from unittest.mock import patch

from line_tracker.best_bets import BetRecommendation
from line_tracker.models import BettingLine, BetType
from line_tracker.slate import (
    _EDGE_OUTLIER_THRESHOLD,
    _STAY_AWAY_LIMIT,
    _TIER1_BASE_EDGE,
    _TIER1_SIGMA_MULT,
    _passes_filters,
    _slate_score,
    _stay_away_sort_key,
    build_daily_slate,
    classify_rec,
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
    edge_z: float = 0.0,
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
        edge_z=edge_z,
    )


def _entry(
    *,
    edge_pct: float = 3.0,
    confidence: str = "High",
    quality_tier: str = "Strong",
    quality_score: int = 80,
    market_volatility_sigma: float = 0.0,
    edge_z: float = 0.0,
    market_unstable: bool = False,
    books_used: int = 5,
    oldest_update_age_min: float = 15.0,
    market: str = "moneyline",
) -> dict:
    """Build a minimal entry dict for classify_rec."""
    return {
        "edge_pct": edge_pct,
        "confidence": confidence,
        "quality_tier": quality_tier,
        "quality_score": quality_score,
        "market_volatility_sigma": market_volatility_sigma,
        "edge_z": edge_z,
        "market_unstable": market_unstable,
        "books_used": books_used,
        "oldest_update_age_min": oldest_update_age_min,
        "market": market,
    }


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
# classify_rec
# ---------------------------------------------------------------------------


class TestClassifyRec:
    def test_tier1(self):
        result = classify_rec(_entry(
            edge_pct=3.0, confidence="High", quality_tier="Strong",
        ))
        assert result["tier"] == "tier1"
        assert result["reasons"] == []

    def test_tier1_elite(self):
        result = classify_rec(_entry(
            edge_pct=4.0, confidence="High", quality_tier="Elite",
            quality_score=90,
        ))
        assert result["tier"] == "tier1"
        assert result["reasons"] == []

    def test_tier2_moderate_quality_medium_conf(self):
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            quality_score=55,
        ))
        assert result["tier"] == "tier2"
        assert result["reasons"] == []

    def test_tier2_high_conf_moderate_tier(self):
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="High", quality_tier="Moderate",
        ))
        assert result["tier"] == "tier2"

    def test_tier2_requires_edge_above_threshold(self):
        result = classify_rec(_entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Moderate",
        ))
        assert result["tier"] == "avoid"
        assert any("Edge too small" in r for r in result["reasons"])

    def test_tier2_edge_z_check(self):
        """Low edge_z prevents Tier 2 even when edge passes."""
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            edge_z=0.5,
        ))
        assert result["tier"] == "avoid"
        assert any("Edge Z-score too low" in r for r in result["reasons"])

    def test_tier2_edge_z_skipped_when_zero(self):
        """edge_z == 0 (unavailable) does not block Tier 2."""
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            edge_z=0.0,
        ))
        assert result["tier"] == "tier2"

    def test_avoid_low_confidence(self):
        result = classify_rec(_entry(confidence="Low"))
        assert result["tier"] == "avoid"
        assert any("Confidence Low" in r for r in result["reasons"])

    def test_avoid_thin_quality(self):
        result = classify_rec(_entry(quality_tier="Thin"))
        assert result["tier"] == "avoid"
        assert any("Quality tier Thin" in r for r in result["reasons"])

    def test_avoid_low_edge(self):
        result = classify_rec(_entry(
            edge_pct=0.5, confidence="Medium", quality_tier="Moderate",
        ))
        assert result["tier"] == "avoid"
        assert any("Edge too small" in r for r in result["reasons"])

    def test_avoid_unstable_market(self):
        result = classify_rec(_entry(market_unstable=True))
        assert result["tier"] == "avoid"
        assert any("Unstable market" in r for r in result["reasons"])

    def test_avoid_too_few_books(self):
        result = classify_rec(_entry(books_used=2))
        assert result["tier"] == "avoid"
        assert any("Too few books" in r for r in result["reasons"])

    def test_avoid_stale_lines(self):
        result = classify_rec(_entry(oldest_update_age_min=200.0))
        assert result["tier"] == "avoid"
        assert any("Stale lines" in r for r in result["reasons"])

    def test_stale_lines_boundary(self):
        """Exactly 120 min does NOT trigger stale-lines reason."""
        result = classify_rec(_entry(oldest_update_age_min=120.0))
        assert not any("Stale lines" in r for r in result["reasons"])
        assert result["tier"] == "tier1"

    def test_avoid_edge_outlier_low_confidence(self):
        result = classify_rec(_entry(
            edge_pct=_EDGE_OUTLIER_THRESHOLD, confidence="Low",
        ))
        assert result["tier"] == "avoid"
        assert any("Edge outlier" in r for r in result["reasons"])

    def test_no_edge_outlier_when_high_confidence(self):
        result = classify_rec(_entry(
            edge_pct=5.0, confidence="High", quality_tier="Strong",
        ))
        assert result["tier"] == "tier1"
        assert not any("Edge outlier" in r for r in result["reasons"])

    def test_tier1_requires_high_confidence(self):
        """Medium confidence falls to Tier 2 (if other Tier 2 criteria met)."""
        result = classify_rec(_entry(
            edge_pct=4.0, confidence="Medium", quality_tier="Strong",
        ))
        assert result["tier"] == "tier2"

    def test_tier1_requires_elite_or_strong_tier(self):
        """Moderate quality_tier falls to Tier 2."""
        result = classify_rec(_entry(
            edge_pct=4.0, confidence="High", quality_tier="Moderate",
        ))
        assert result["tier"] == "tier2"

    def test_exactly_4_books_no_too_few_reason(self):
        """Exactly 4 books does NOT trigger the too-few-books reason."""
        result = classify_rec(_entry(books_used=4))
        assert not any("Too few books" in r for r in result["reasons"])
        assert result["tier"] == "tier1"

    def test_avoid_multiple_reasons(self):
        result = classify_rec(_entry(
            edge_pct=0.5, confidence="Low", quality_tier="Thin",
            market_unstable=True, books_used=2, oldest_update_age_min=200.0,
        ))
        assert result["tier"] == "avoid"
        # Hard disqualifiers: unstable, too few books, stale lines
        assert len(result["reasons"]) >= 3

    def test_dynamic_edge_floor_returned(self):
        result = classify_rec(_entry(market_volatility_sigma=0.5))
        expected = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * 0.5
        assert result["dynamic_edge_floor"] == expected

    def test_tier1_dynamic_floor_fail_gives_reason(self):
        """When a Strong/High rec just misses Tier 1's dynamic floor."""
        sigma = 0.5
        floor = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma
        result = classify_rec(_entry(
            edge_pct=floor - 0.1, confidence="High", quality_tier="Strong",
            market_volatility_sigma=sigma,
        ))
        # Edge is above _TIER2_EDGE so it should land in Tier 2
        assert result["tier"] == "tier2"

    def test_avoid_includes_dynamic_floor_reason_when_below_tier2(self):
        """Strong/High rec with edge below Tier 2 minimum gets floor reason."""
        sigma = 0.5
        result = classify_rec(_entry(
            edge_pct=1.0, confidence="High", quality_tier="Strong",
            market_volatility_sigma=sigma,
        ))
        assert result["tier"] == "avoid"
        assert any("Edge too small" in r for r in result["reasons"])
        assert any("Fails Tier 1 dynamic floor" in r for r in result["reasons"])

    def test_edge_not_positive(self):
        result = classify_rec(_entry(edge_pct=0.0))
        assert result["tier"] == "avoid"
        assert any("Edge not positive" in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# _passes_filters
# ---------------------------------------------------------------------------


class TestPassesFilters:
    def _fentry(self, **overrides):
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
        assert _passes_filters(self._fentry(), {}) is True

    def test_min_edge_pass(self):
        assert _passes_filters(self._fentry(edge_pct=3.0), {"min_edge": 2.0}) is True

    def test_min_edge_fail(self):
        assert _passes_filters(self._fentry(edge_pct=1.0), {"min_edge": 2.0}) is False

    def test_min_quality_pass(self):
        e = self._fentry(quality_score=80)
        assert _passes_filters(e, {"min_quality": 60}) is True

    def test_min_quality_fail(self):
        e = self._fentry(quality_score=30)
        assert _passes_filters(e, {"min_quality": 60}) is False

    def test_markets_pass(self):
        assert _passes_filters(
            self._fentry(market="spread"), {"markets": ["spread", "total"]}
        ) is True

    def test_markets_fail(self):
        assert _passes_filters(
            self._fentry(market="moneyline"), {"markets": ["spread", "total"]}
        ) is False

    def test_hide_low_confidence_pass(self):
        assert _passes_filters(
            self._fentry(confidence="High"), {"hide_low_confidence": True}
        ) is True

    def test_hide_low_confidence_fail(self):
        assert _passes_filters(
            self._fentry(confidence="Low"), {"hide_low_confidence": True}
        ) is False

    def test_books_used_min_pass(self):
        e = self._fentry(books_used=5)
        assert _passes_filters(e, {"books_used_min": 3}) is True

    def test_books_used_min_fail(self):
        e = self._fentry(books_used=2)
        assert _passes_filters(e, {"books_used_min": 3}) is False

    def test_combined_filters(self):
        entry = self._fentry(edge_pct=4.0, quality_score=90, market="spread")
        filters = {"min_edge": 2.0, "min_quality": 50, "markets": ["spread"]}
        assert _passes_filters(entry, filters) is True

    def test_combined_filters_one_fails(self):
        entry = self._fentry(edge_pct=1.0, quality_score=90, market="spread")
        filters = {"min_edge": 2.0, "min_quality": 50, "markets": ["spread"]}
        assert _passes_filters(entry, filters) is False


# ---------------------------------------------------------------------------
# _stay_away_sort_key
# ---------------------------------------------------------------------------


class TestStayAwaySortKey:
    def test_low_confidence_sorts_first(self):
        low = _entry(confidence="Low", quality_score=50)
        high = _entry(confidence="High", quality_score=50)
        assert _stay_away_sort_key(low) < _stay_away_sort_key(high)

    def test_lower_edge_z_sorts_first(self):
        bad = _entry(edge_z=-1.0, quality_score=50)
        ok = _entry(edge_z=0.5, quality_score=50)
        assert _stay_away_sort_key(bad) < _stay_away_sort_key(ok)

    def test_higher_sigma_sorts_first(self):
        vol = _entry(market_volatility_sigma=2.0, quality_score=50)
        calm = _entry(market_volatility_sigma=0.01, quality_score=50)
        assert _stay_away_sort_key(vol) < _stay_away_sort_key(calm)


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
            quality_score=55, edge_pct=2.0,
            quality_tier="Moderate", confidence="Medium",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["tier2"]) == 1
        assert result["tier2"][0]["tier"] == "tier2"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_basic_avoid(self, mock_rbb):
        """A low-quality recommendation lands in avoid."""
        mock_rbb.return_value = [_make_rec(
            quality_score=20, edge_pct=0.5, quality_tier="Thin",
        )]
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
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"max_per_event": 2},
        )
        total = len(result["tier1"]) + len(result["tier2"]) + len(result["avoid"])
        assert total == 2

    @patch("line_tracker.slate.recommend_best_bets")
    def test_filter_min_edge_does_not_drop_avoid(self, mock_rbb):
        """min_edge display-filter does NOT prevent classification or avoid."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=1.0, quality_tier="Strong",
            confidence="High",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"min_edge": 2.0}
        )
        # Entry is classified as avoid (edge too small for Tier 2),
        # and NOT dropped by the display filter.
        assert len(result["avoid"]) == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_filter_markets(self, mock_rbb):
        """Only specified markets survive (applies to all tiers)."""
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
    def test_filter_hide_low_confidence_does_not_drop_avoid(self, mock_rbb):
        """hide_low_confidence does NOT suppress Stay Away entries."""
        mock_rbb.return_value = [_make_rec(
            confidence="Low", quality_score=80, edge_pct=3.0,
            quality_tier="Thin",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"hide_low_confidence": True}
        )
        # Low confidence recs are classified to avoid, not dropped
        assert len(result["avoid"]) == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_empty_input(self, mock_rbb):
        """Empty lines_by_event produces empty tiers."""
        result = build_daily_slate({})
        assert result["tier1"] == []
        assert result["tier2"] == []
        assert result["avoid"] == []
        mock_rbb.assert_not_called()

    @patch("line_tracker.slate.recommend_best_bets")
    def test_no_recommendations(self, mock_rbb):
        """Event with no recommendations is skipped."""
        mock_rbb.return_value = []
        result = build_daily_slate({"evt1": _event_lines()})
        assert result["tier1"] == []
        assert result["tier2"] == []
        assert result["avoid"] == []

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
                    quality_score=55, edge_pct=2.0,
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

    @patch("line_tracker.slate.recommend_best_bets")
    def test_stay_away_capped_at_limit(self, mock_rbb):
        """Stay Away is capped at _STAY_AWAY_LIMIT entries."""

        def side_effect(lines):
            return [_make_rec(
                quality_score=20, edge_pct=0.3,
                quality_tier="Thin", confidence="Low",
            )]

        mock_rbb.side_effect = side_effect
        # Create more events than the limit
        lines = {
            f"evt{i}": _event_lines(f"Team{i} @ Team{i+100}")
            for i in range(_STAY_AWAY_LIMIT + 5)
        }
        result = build_daily_slate(lines)
        assert len(result["avoid"]) == _STAY_AWAY_LIMIT

    @patch("line_tracker.slate.recommend_best_bets")
    def test_debug_counters_when_enabled(self, mock_rbb):
        """Debug counters are present when debug flag is set."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong", confidence="High",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"debug": True}
        )
        debug = result["debug"]
        assert debug["total_recs"] == 1
        assert debug["tier1_count"] == 1
        assert debug["tier2_count"] == 0
        assert debug["stay_away_count"] == 0
        assert "High" in debug["by_confidence"]
        assert "Strong" in debug["by_quality_tier"]

    @patch("line_tracker.slate.recommend_best_bets")
    def test_no_debug_by_default(self, mock_rbb):
        """Debug counters are absent by default."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert "debug" not in result


# ---------------------------------------------------------------------------
# Key scenario: recs exist but none meet Tier1/Tier2 → Stay Away non-empty
# ---------------------------------------------------------------------------


class TestStayAwayAlwaysPopulated:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_all_recs_to_stay_away(self, mock_rbb):
        """When all recs fail Tier1/Tier2, Stay Away is non-empty with reasons."""

        def side_effect(lines):
            return [_make_rec(
                quality_score=30, edge_pct=0.3,
                quality_tier="Thin", confidence="Low",
            )]

        mock_rbb.side_effect = side_effect
        lines = {
            "evt1": _event_lines("Knicks @ Heat"),
            "evt2": _event_lines("Celtics @ Bucks"),
        }
        result = build_daily_slate(lines)
        assert len(result["tier1"]) == 0
        assert len(result["tier2"]) == 0
        assert len(result["avoid"]) == 2
        for entry in result["avoid"]:
            assert len(entry["avoid_reasons"]) >= 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_stay_away_not_suppressed_by_display_filters(self, mock_rbb):
        """Display filters (min_edge, min_quality, hide_low) don't suppress avoid."""
        mock_rbb.return_value = [_make_rec(
            quality_score=30, edge_pct=0.3,
            quality_tier="Thin", confidence="Low",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()},
            filters={"min_edge": 5.0, "min_quality": 90, "hide_low_confidence": True},
        )
        assert len(result["avoid"]) == 1
        assert len(result["avoid"][0]["avoid_reasons"]) >= 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_medium_quality_recs_go_to_stay_away_with_reasons(self, mock_rbb):
        """Recs with moderate stats that miss Tier 2 edge get Stay Away + reasons."""
        mock_rbb.return_value = [_make_rec(
            quality_score=55, edge_pct=1.0,
            quality_tier="Moderate", confidence="Medium",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["avoid"]) == 1
        entry = result["avoid"][0]
        assert any("Edge too small" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_entry_carries_edge_z_and_market_unstable(self, mock_rbb):
        """Entries include edge_z and market_unstable fields."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong",
            confidence="High", edge_z=2.5, market_unstable=False,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["tier1"][0]
        assert entry["edge_z"] == 2.5
        assert entry["market_unstable"] is False


# ---------------------------------------------------------------------------
# Dynamic edge floor tests
# ---------------------------------------------------------------------------


class TestDynamicEdgeFloor:
    def test_zero_sigma_uses_base_edge(self):
        """With sigma=0, dynamic floor equals the base edge (3.0%)."""
        result = classify_rec(_entry(
            edge_pct=_TIER1_BASE_EDGE, confidence="High",
            quality_tier="Strong", market_volatility_sigma=0.0,
        ))
        assert result["tier"] == "tier1"

    def test_below_base_edge_falls_to_tier2(self):
        """Edge below 3.0% base with zero sigma → tier2 (if Tier 2 criteria met)."""
        result = classify_rec(_entry(
            edge_pct=2.99, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.0,
        ))
        # 2.99 >= _TIER2_EDGE (1.5) → tier2
        assert result["tier"] == "tier2"

    def test_sigma_raises_floor(self):
        """sigma=0.02 → floor=3.024; edge=3.01 < 3.024 → not tier1."""
        result = classify_rec(_entry(
            edge_pct=3.01, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.02,
        ))
        # Misses Tier 1, but 3.01 >= 1.5 → tier2
        assert result["tier"] == "tier2"

    def test_edge_above_raised_floor(self):
        """sigma=0.02 → floor = 3.024; edge=3.03 >= 3.024 → tier1."""
        result = classify_rec(_entry(
            edge_pct=3.03, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.02,
        ))
        assert result["tier"] == "tier1"

    def test_high_sigma_needs_large_edge(self):
        """sigma=1.0 → floor = 4.2; edge=4.0 < 4.2 → not tier1."""
        result = classify_rec(_entry(
            edge_pct=4.0, confidence="High", quality_tier="Strong",
            market_volatility_sigma=1.0,
        ))
        # 4.0 >= 1.5 → tier2
        assert result["tier"] == "tier2"

    def test_high_sigma_edge_above(self):
        """sigma=1.0 → floor = 4.2; edge=4.3 >= 4.2 → tier1."""
        result = classify_rec(_entry(
            edge_pct=4.3, confidence="High", quality_tier="Strong",
            market_volatility_sigma=1.0,
        ))
        assert result["tier"] == "tier1"

    def test_dynamic_floor_formula(self):
        """Verify the formula: floor = base + mult * sigma."""
        sigma = 0.05
        expected_floor = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma
        # edge just at the floor → tier1
        r_at = classify_rec(_entry(
            edge_pct=expected_floor, confidence="High",
            quality_tier="Strong", market_volatility_sigma=sigma,
        ))
        assert r_at["tier"] == "tier1"
        # edge just below the floor → not tier1 (tier2 since >= 1.5)
        r_below = classify_rec(_entry(
            edge_pct=expected_floor - 0.001, confidence="High",
            quality_tier="Strong", market_volatility_sigma=sigma,
        ))
        assert r_below["tier"] == "tier2"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_dynamic_floor_integration(self, mock_rbb):
        """Integration: high sigma pushes a borderline rec from tier1 to tier2."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.1,
            quality_tier="Strong", confidence="High",
            market_volatility_sigma=0.1,  # floor = 3.0 + 1.2*0.1 = 3.12
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        # 3.1 < 3.12 → tier2 (since 3.1 >= 1.5)
        assert len(result["tier2"]) == 1
        assert len(result["tier1"]) == 0

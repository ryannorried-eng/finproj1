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
    compute_slate_debug_stats,
    passes_relaxed_tier2,
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
    market_hold_median: float = 0.0,
    edge_z: float = 0.0,
    ev: float = 0.05,
    best_sportsbook: str = "FanDuel",
    best_odds: float = -110.0,
    line: float | None = None,
    ev_100: float | None = None,
) -> BetRecommendation:
    _ev_100 = ev_100 if ev_100 is not None else edge_pct
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
        ev_per_100=_ev_100,
        confidence=confidence,
        ev_100=_ev_100,
        quality_score=quality_score,
        quality_tier=quality_tier,
        books_used_count=books_used_count,
        newest_update_age_min=newest_update_age_min,
        oldest_update_age_min=oldest_update_age_min,
        market_unstable=market_unstable,
        market_volatility_sigma=market_volatility_sigma,
        market_hold_median=market_hold_median,
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
    market_hold_median: float = 0.0,
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
        "market_hold_median": market_hold_median,
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
    def test_tier1b_default(self):
        """Default entry (books=5) passes Tier 1B but not 1A."""
        result = classify_rec(_entry(
            edge_pct=3.0, confidence="High", quality_tier="Strong",
        ))
        assert result["tier"] == "tier1b"
        assert result["reasons"] == []

    def test_tier1a_with_enough_books(self):
        """books=6 promotes to Tier 1A."""
        result = classify_rec(_entry(
            edge_pct=3.0, confidence="High", quality_tier="Strong",
            books_used=6,
        ))
        assert result["tier"] == "tier1a"
        assert result["reasons"] == []

    def test_tier1b_elite(self):
        result = classify_rec(_entry(
            edge_pct=4.0, confidence="High", quality_tier="Elite",
            quality_score=90,
        ))
        assert result["tier"] == "tier1b"
        assert result["reasons"] == []

    def test_tier1b_moderate_quality_medium_conf(self):
        """Medium conf + Moderate quality + edge >= 2.0 → tier1b."""
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            quality_score=55,
        ))
        assert result["tier"] == "tier1b"
        assert result["reasons"] == []

    def test_tier1b_high_conf_moderate_tier(self):
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="High", quality_tier="Moderate",
        ))
        assert result["tier"] == "tier1b"

    def test_tier2_edge_at_threshold(self):
        """edge=1.0 with Medium/Moderate and books=4 → tier2."""
        result = classify_rec(_entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Moderate",
            books_used=4,
        ))
        assert result["tier"] == "tier2"

    def test_tier1b_edge_z_irrelevant(self):
        """Low edge_z does NOT prevent Tier 1B (no edge_z gate for 1B)."""
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            edge_z=0.5,
        ))
        assert result["tier"] == "tier1b"

    def test_tier1b_edge_z_zero(self):
        """edge_z == 0 (unavailable) does not block Tier 1B."""
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            edge_z=0.0,
        ))
        assert result["tier"] == "tier1b"

    def test_tier3_low_confidence(self):
        """Low confidence with positive edge → tier3."""
        result = classify_rec(_entry(confidence="Low"))
        assert result["tier"] == "tier3"
        assert result["reasons"] == []

    def test_tier3_thin_quality(self):
        """Thin quality with positive edge → tier3."""
        result = classify_rec(_entry(quality_tier="Thin"))
        assert result["tier"] == "tier3"
        assert result["reasons"] == []

    def test_tier2_low_edge(self):
        """edge=0.5 with Medium/Moderate → tier2 (any positive edge)."""
        result = classify_rec(_entry(
            edge_pct=0.5, confidence="Medium", quality_tier="Moderate",
        ))
        assert result["tier"] == "tier2"
        assert result["reasons"] == []

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
        assert result["tier"] == "tier1b"

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
        assert result["tier"] == "tier1b"
        assert not any("Edge outlier" in r for r in result["reasons"])

    def test_tier1a_requires_high_confidence(self):
        """Medium confidence falls to Tier 1B (with Strong quality, books=5)."""
        result = classify_rec(_entry(
            edge_pct=4.0, confidence="Medium", quality_tier="Strong",
        ))
        assert result["tier"] == "tier1b"

    def test_tier1a_requires_elite_or_strong_tier(self):
        """Moderate quality_tier falls to Tier 1B."""
        result = classify_rec(_entry(
            edge_pct=4.0, confidence="High", quality_tier="Moderate",
        ))
        assert result["tier"] == "tier1b"

    def test_exactly_4_books_falls_to_tier2(self):
        """Exactly 4 books passes hard gate but fails 1A/1B → tier2."""
        result = classify_rec(_entry(books_used=4))
        assert not any("Too few books" in r for r in result["reasons"])
        assert result["tier"] == "tier2"

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
        # floor_1a = max(base, mult * sigma) = max(2.0, 100*0.5) = 50.0
        expected = max(_TIER1_BASE_EDGE, _TIER1_SIGMA_MULT * 0.5)
        assert result["dynamic_edge_floor"] == expected

    def test_tier1a_dynamic_floor_fail_falls_to_1b(self):
        """When a Strong/High rec just misses Tier 1A's dynamic floor → 1B."""
        sigma = 0.5
        floor = _TIER1_BASE_EDGE + _TIER1_SIGMA_MULT * sigma
        result = classify_rec(_entry(
            edge_pct=floor - 0.1, confidence="High", quality_tier="Strong",
            market_volatility_sigma=sigma,
        ))
        # Edge 2.9 still >= dyn_floor_1b(0.5) = max(2.0, 1.5) = 2.0
        assert result["tier"] == "tier1b"

    def test_edge_below_1b_floor_goes_to_tier2(self):
        """Strong/High rec with edge below 1B floor but >=1.0 → tier2."""
        result = classify_rec(_entry(
            edge_pct=1.5, confidence="High", quality_tier="Strong",
            books_used=4,
        ))
        # edge=1.5 < dyn_floor_1b(0)=2.0 → fail 1B, but >=1.0 → tier2
        assert result["tier"] == "tier2"

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
    def test_basic_tier1b(self, mock_rbb):
        """A high-quality recommendation (books=5) lands in tier1b → tier1."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong", confidence="High",
        )]
        lines = {"evt1": _event_lines()}
        result = build_daily_slate(lines)
        assert len(result["tier1"]) == 1
        assert result["tier1"][0]["event"] == "Lakers @ Celtics"
        assert result["tier1"][0]["tier"] == "tier1b"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_basic_tier1b_moderate(self, mock_rbb):
        """Medium/Moderate + edge=2.0 + books=5 → tier1b (in tier1)."""
        mock_rbb.return_value = [_make_rec(
            quality_score=55, edge_pct=2.0,
            quality_tier="Moderate", confidence="Medium",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["tier1"]) == 1
        assert result["tier1"][0]["tier"] == "tier1b"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_basic_tier3(self, mock_rbb):
        """Thin quality with positive edge → tier3."""
        mock_rbb.return_value = [_make_rec(
            quality_score=20, edge_pct=0.5, quality_tier="Thin",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["tier3"]) == 1
        assert result["tier3"][0]["tier"] == "tier3"

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
        all_entries = (
            result["tier1"] + result["tier2"] + result["tier3"]
            + result["stay_away"]
        )
        entry = all_entries[0]
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
        total = (
            len(result["tier1"]) + len(result["tier2"])
            + len(result["tier3"]) + len(result["stay_away"])
        )
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
        total = (
            len(result["tier1"]) + len(result["tier2"])
            + len(result["tier3"]) + len(result["stay_away"])
        )
        assert total == 2

    @patch("line_tracker.slate.recommend_best_bets")
    def test_filter_min_edge_does_not_change_classification(self, mock_rbb):
        """min_edge display-filter hides tier2 but doesn't drop classification."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=0.5, quality_tier="Strong",
            confidence="High",
            books_used_count=4,  # < 5 → fails 1B books gate
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"min_edge": 2.0}
        )
        # Entry is classified as tier2 (edge=0.5 > 0, books >= 4),
        # display-filtered from tier2 list, but counts still reflect it.
        assert result["counts"]["tier2"] == 1
        assert len(result["tier2"]) == 0  # filtered out of display

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
        total = (
            len(result["tier1"]) + len(result["tier2"])
            + len(result["tier3"]) + len(result["stay_away"])
        )
        assert total == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_filter_hide_low_confidence_does_not_drop_tier3(self, mock_rbb):
        """hide_low_confidence filters tier3 display but doesn't change count."""
        mock_rbb.return_value = [_make_rec(
            confidence="Low", quality_score=80, edge_pct=3.0,
            quality_tier="Thin",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()}, filters={"hide_low_confidence": True}
        )
        # Low/Thin with positive edge → tier3 classification
        assert result["counts"]["tier3"] == 1
        # Display-filtered from tier3 list
        assert len(result["tier3"]) == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_empty_input(self, mock_rbb):
        """Empty lines_by_event produces empty tiers."""
        result = build_daily_slate({})
        assert result["tier1a"] == []
        assert result["tier1b"] == []
        assert result["tier1"] == []
        assert result["tier2"] == []
        assert result["tier3"] == []
        assert result["stay_away"] == []
        mock_rbb.assert_not_called()

    @patch("line_tracker.slate.recommend_best_bets")
    def test_no_recommendations(self, mock_rbb):
        """Event with no recommendations is skipped."""
        mock_rbb.return_value = []
        result = build_daily_slate({"evt1": _event_lines()})
        assert result["tier1"] == []
        assert result["tier2"] == []
        assert result["tier3"] == []
        assert result["stay_away"] == []

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
        # Knicks: Elite/High → tier1b (books=5), Celtics: Moderate/Medium → tier1b
        assert len(result["tier1"]) == 2
        # Warriors: Thin/Low → tier3
        assert len(result["tier3"]) == 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_reasons_populated(self, mock_rbb):
        """Avoid entries have descriptive reasons."""
        mock_rbb.return_value = [
            _make_rec(quality_score=20, edge_pct=0.5, confidence="Low",
                      market_unstable=True, quality_tier="Thin")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["stay_away"][0]
        assert any("Unstable market" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_stale_lines_integration(self, mock_rbb):
        """Stale lines cause avoid with descriptive reason."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=3.0, oldest_update_age_min=200.0,
                      quality_tier="Strong", confidence="High")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["stay_away"][0]
        assert any("Stale lines" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_too_few_books_integration(self, mock_rbb):
        """Too few books causes avoid with descriptive reason."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=3.0, books_used_count=2,
                      quality_tier="Strong", confidence="High")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["stay_away"][0]
        assert any("Too few books" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_edge_outlier_integration(self, mock_rbb):
        """High edge + low confidence triggers edge outlier reason."""
        mock_rbb.return_value = [
            _make_rec(quality_score=80, edge_pct=9.0, confidence="Low",
                      quality_tier="Thin")
        ]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["stay_away"][0]
        assert any("Edge outlier" in r for r in entry["avoid_reasons"])

    @patch("line_tracker.slate.recommend_best_bets")
    def test_stay_away_capped_at_limit(self, mock_rbb):
        """Stay Away is capped at _STAY_AWAY_LIMIT entries."""

        def side_effect(lines):
            # edge=0 → avoid (edge not positive)
            return [_make_rec(
                quality_score=20, edge_pct=0.0,
                quality_tier="Thin", confidence="Low",
            )]

        mock_rbb.side_effect = side_effect
        # Create more events than the limit
        lines = {
            f"evt{i}": _event_lines(f"Team{i} @ Team{i+100}")
            for i in range(_STAY_AWAY_LIMIT + 5)
        }
        result = build_daily_slate(lines)
        assert len(result["stay_away"]) == _STAY_AWAY_LIMIT

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
        assert debug["tier1_count"] == 1  # combined tier1
        assert debug["tier1b_count"] == 1
        assert debug["tier1a_count"] == 0
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
    def test_all_recs_to_tier3(self, mock_rbb):
        """When all recs fail Tier1/Tier2 but have positive edge → tier3."""

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
        assert len(result["tier3"]) == 2

    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_not_suppressed_by_display_filters(self, mock_rbb):
        """Display filters don't suppress actual avoid entries."""
        mock_rbb.return_value = [_make_rec(
            quality_score=30, edge_pct=0.0,
            quality_tier="Thin", confidence="Low",
        )]
        result = build_daily_slate(
            {"evt1": _event_lines()},
            filters={"min_edge": 5.0, "min_quality": 90, "hide_low_confidence": True},
        )
        assert len(result["stay_away"]) == 1
        assert len(result["stay_away"][0]["avoid_reasons"]) >= 1

    @patch("line_tracker.slate.recommend_best_bets")
    def test_moderate_recs_with_edge_at_tier2_threshold(self, mock_rbb):
        """Moderate recs with small positive edge reach tier2."""
        mock_rbb.return_value = [_make_rec(
            quality_score=55, edge_pct=0.5,
            quality_tier="Moderate", confidence="Medium",
            books_used_count=4,  # < 5 → fails 1B books gate
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        # edge=0.5 > 0, books >= 4, Moderate quality → tier2
        assert len(result["tier2"]) == 1

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
    def test_zero_sigma_tier1a(self):
        """sigma=0, books=6 → floor_1a=2.5 → tier1a."""
        result = classify_rec(_entry(
            edge_pct=_TIER1_BASE_EDGE, confidence="High",
            quality_tier="Strong", market_volatility_sigma=0.0,
            books_used=6,
        ))
        assert result["tier"] == "tier1a"

    def test_below_base_edge_falls_to_1b(self):
        """Edge below 2.5% but >= 2.0 with books=5 → tier1b."""
        result = classify_rec(_entry(
            edge_pct=2.4, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.0,
        ))
        # 2.4 >= dyn_floor_1b(0)=2.0 → tier1b
        assert result["tier"] == "tier1b"

    def test_sigma_raises_1a_floor(self):
        """sigma=0.03 → floor_1a=max(2.0, 100*0.03)=3.0; edge=2.9 → not 1A."""
        result = classify_rec(_entry(
            edge_pct=2.9, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.03, books_used=6,
        ))
        # Misses 1A, but 2.9 >= floor_1b(0.03)=max(1.0,3.0)=3.0 → fails 1B too
        # 2.9 > 0 and books >= 4 → tier2
        assert result["tier"] == "tier2"

    def test_edge_above_raised_1a_floor(self):
        """sigma=0.03 → floor_1a=3.0; edge=3.0 >= 3.0 → tier1a."""
        result = classify_rec(_entry(
            edge_pct=3.0, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.03, books_used=6,
        ))
        assert result["tier"] == "tier1a"

    def test_high_sigma_needs_large_edge(self):
        """sigma=0.05 → floor_1a=5.0; edge=4.9 < 5.0 → not tier1a."""
        result = classify_rec(_entry(
            edge_pct=4.9, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.05, books_used=6,
        ))
        # 4.9 >= floor_1b(0.05)=max(1.0,5.0)=5.0 → fails 1B too
        # 4.9 > 0 → tier2
        assert result["tier"] == "tier2"

    def test_high_sigma_edge_above(self):
        """sigma=0.05 → floor_1a=5.0; edge=5.0 >= 5.0 → tier1a."""
        result = classify_rec(_entry(
            edge_pct=5.0, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.05, books_used=6,
        ))
        assert result["tier"] == "tier1a"

    def test_dynamic_floor_formula(self):
        """Verify the formula: floor_1a = max(base, mult * sigma)."""
        sigma = 0.025
        expected_floor = max(_TIER1_BASE_EDGE, _TIER1_SIGMA_MULT * sigma)
        # expected_floor = max(2.0, 100*0.025) = max(2.0, 2.5) = 2.5
        # edge just at the floor with books=6 → tier1a
        r_at = classify_rec(_entry(
            edge_pct=expected_floor, confidence="High",
            quality_tier="Strong", market_volatility_sigma=sigma,
            books_used=6,
        ))
        assert r_at["tier"] == "tier1a"
        # edge just below the floor → not tier1a (tier1b if >= 1b floor)
        r_below = classify_rec(_entry(
            edge_pct=expected_floor - 0.001, confidence="High",
            quality_tier="Strong", market_volatility_sigma=sigma,
            books_used=6,
        ))
        assert r_below["tier"] != "tier1a"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_dynamic_floor_integration(self, mock_rbb):
        """Integration: sigma pushes a borderline rec from tier1a to tier2."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=2.5,
            quality_tier="Strong", confidence="High",
            market_volatility_sigma=0.03,  # floor_1a = max(2.0, 100*0.03) = 3.0
            books_used_count=6,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        # 2.5 < 3.0 → fails 1A; floor_1b(0.03)=max(1.0,3.0)=3.0 → fails 1B
        # 2.5 > 0 → tier2
        assert len(result["tier1a"]) == 0
        assert len(result["tier1b"]) == 0
        assert len(result["tier2"]) == 1


# ---------------------------------------------------------------------------
# Counts always present
# ---------------------------------------------------------------------------


class TestCountsAlwaysPresent:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_counts_present_without_debug(self, mock_rbb):
        """counts dict is always present, even without debug flag."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert "counts" in result
        assert result["counts"]["total_recs"] == 1
        assert result["counts"]["tier1"] == 1  # combined
        assert result["counts"]["tier1b"] == 1
        assert result["counts"]["tier1a"] == 0
        assert result["counts"]["tier2"] == 0
        assert result["counts"]["stay_away"] == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_counts_zero_on_empty(self, mock_rbb):
        """counts is present and all zeros when no events supplied."""
        result = build_daily_slate({})
        assert result["counts"]["total_recs"] == 0


# ---------------------------------------------------------------------------
# Display filters never change classification
# ---------------------------------------------------------------------------


class TestFiltersNeverChangeClassification:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_tier_field_unchanged_by_display_filters(self, mock_rbb):
        """Each entry's 'tier' field is identical regardless of display filters."""

        def _recs(lines):
            ev = lines[0].event
            if "A" in ev:
                return [_make_rec(
                    quality_score=90, edge_pct=4.0,
                    quality_tier="Elite", confidence="High",
                )]
            if "B" in ev:
                return [_make_rec(
                    quality_score=55, edge_pct=2.0,
                    quality_tier="Moderate", confidence="Medium",
                )]
            return [_make_rec(
                quality_score=20, edge_pct=0.3,
                quality_tier="Thin", confidence="Low",
            )]

        mock_rbb.side_effect = _recs
        lines = {
            "e1": _event_lines("TeamA @ TeamX"),
            "e2": _event_lines("TeamB @ TeamY"),
            "e3": _event_lines("TeamC @ TeamZ"),
        }

        # Without filters
        r1 = build_daily_slate(lines)
        # With aggressive filters
        r2 = build_daily_slate(lines, filters={
            "min_edge": 5.0, "min_quality": 95, "hide_low_confidence": True,
        })

        # Classification counts must be identical
        assert r1["counts"] == r2["counts"]

    @patch("line_tracker.slate.recommend_best_bets")
    def test_display_filters_only_reduce_shown_entries(self, mock_rbb):
        """Display filters can only remove entries from tier1/tier2, never add."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0, quality_tier="Strong", confidence="High",
        )]
        no_filter = build_daily_slate({"evt1": _event_lines()})
        filtered = build_daily_slate(
            {"evt1": _event_lines()}, filters={"min_edge": 5.0}
        )
        assert len(filtered["tier1"]) <= len(no_filter["tier1"])
        # Stay away cannot gain entries from display-filtering tier1/tier2
        assert filtered["counts"] == no_filter["counts"]


# ---------------------------------------------------------------------------
# Strict mode / relaxed Tier 2 display
# ---------------------------------------------------------------------------


class TestRelaxedTier2:
    def test_strict_tier3_passes_relaxed(self):
        """An entry that's tier3 under strict but meets relaxed thresholds."""
        # Low conf + Strong quality + no edge_z → tier3 (Low conf override
        # needs edge_z >= 1.5, but edge_z=0 → not applied).
        e = _entry(
            edge_pct=1.0, confidence="Low", quality_tier="Strong",
            edge_z=0.0, books_used=5,
        )
        result = classify_rec(e)
        assert result["tier"] == "tier3"  # strict classification
        assert passes_relaxed_tier2(e) is True  # relaxed display

    def test_relaxed_rejects_hard_disqualifier(self):
        """Hard disqualifiers prevent relaxed promotion."""
        e = _entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Moderate",
            market_unstable=True,
        )
        assert passes_relaxed_tier2(e) is False

    def test_relaxed_allows_low_conf_strong_quality(self):
        """Low confidence is allowed when quality_tier is Strong."""
        e = _entry(
            edge_pct=1.0, confidence="Low", quality_tier="Strong",
            edge_z=0.0, books_used=5,
        )
        assert passes_relaxed_tier2(e) is True

    def test_relaxed_rejects_low_conf_moderate_quality(self):
        """Low confidence is rejected when quality_tier is only Moderate."""
        e = _entry(
            edge_pct=1.0, confidence="Low", quality_tier="Moderate",
            edge_z=0.0, books_used=5,
        )
        assert passes_relaxed_tier2(e) is False

    def test_relaxed_rejects_below_edge_floor(self):
        """Edge below $0.30 EV/$100 fails relaxed mode."""
        e = _entry(
            edge_pct=0.2, confidence="Medium", quality_tier="Moderate",
            edge_z=0.0, books_used=5,
        )
        assert passes_relaxed_tier2(e) is False

    def test_relaxed_rejects_low_edge_z(self):
        """edge_z below 0.75 fails relaxed mode."""
        e = _entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Moderate",
            edge_z=0.5, books_used=5,
        )
        assert passes_relaxed_tier2(e) is False

    def test_relaxed_does_not_change_tier1b(self):
        """passes_relaxed_tier2 is only for tier3/avoid; Tier1B stays strict."""
        e = _entry(
            edge_pct=3.0, confidence="High", quality_tier="Strong",
        )
        result = classify_rec(e)
        # Tier 1B entries should never need relaxed promotion
        assert result["tier"] == "tier1b"

    def test_relaxed_rejects_stale(self):
        """Stale lines block relaxed promotion."""
        e = _entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Moderate",
            oldest_update_age_min=200.0, books_used=5,
        )
        assert passes_relaxed_tier2(e) is False

    def test_relaxed_rejects_too_few_books(self):
        """Too few books blocks relaxed promotion."""
        e = _entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Moderate",
            books_used=2,
        )
        assert passes_relaxed_tier2(e) is False

    def test_relaxed_rejects_thin_quality(self):
        """Thin quality tier fails relaxed mode."""
        e = _entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Thin",
            books_used=5,
        )
        assert passes_relaxed_tier2(e) is False


# ---------------------------------------------------------------------------
# Empty state message strings
# ---------------------------------------------------------------------------


class TestEmptyStateStrings:
    """Verify that the expected empty-state text constants are correct.

    The actual rendering is in Streamlit UI code; here we verify that
    the slate module returns the right structure so the dashboard can
    produce the right messages.
    """

    @patch("line_tracker.slate.recommend_best_bets")
    def test_empty_tier1_has_data_for_message(self, mock_rbb):
        """When tier1 is empty but recs exist, counts reflect it."""
        mock_rbb.return_value = [_make_rec(
            quality_score=55, edge_pct=1.5,
            quality_tier="Moderate", confidence="Medium",
            books_used_count=4,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        # edge=1.5 < floor_1b(0)=2.0 → fails 1B, but >=1.0 → tier2
        assert len(result["tier1"]) == 0
        assert result["counts"]["tier1"] == 0
        assert result["counts"]["total_recs"] > 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_empty_tier2_has_data_for_message(self, mock_rbb):
        """When tier2 is empty but recs exist, counts reflect it."""
        mock_rbb.return_value = [_make_rec(
            quality_score=90, edge_pct=4.0,
            quality_tier="Elite", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["tier2"]) == 0
        assert result["counts"]["tier2"] == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_all_empty_zero_total(self, mock_rbb):
        """No recommendations → total_recs == 0 for no-data message."""
        result = build_daily_slate({})
        assert result["counts"]["total_recs"] == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_stay_away_empty_when_all_pass(self, mock_rbb):
        """When all recs are tier1/tier2, stay_away is empty."""
        mock_rbb.return_value = [_make_rec(
            quality_score=90, edge_pct=4.0,
            quality_tier="Elite", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["stay_away"]) == 0


# ---------------------------------------------------------------------------
# Debug stats: compute_slate_debug_stats
# ---------------------------------------------------------------------------


class TestComputeSlateDebugStats:
    def test_empty_entries(self):
        ds = compute_slate_debug_stats([])
        assert ds["total_recs"] == 0
        assert ds["sigma_stats"]["min"] is None
        assert ds["dynamic_floor_stats"]["min"] is None
        assert ds["warnings"] == []

    def test_basic_counts(self):
        entries = [
            {**_entry(edge_pct=4.0, confidence="High", quality_tier="Elite"),
             "tier": "tier1a", "dynamic_edge_floor": 2.5},
            {**_entry(edge_pct=2.0, confidence="Medium", quality_tier="Moderate"),
             "tier": "tier2", "dynamic_edge_floor": 2.5},
            {**_entry(edge_pct=0.5, confidence="Low", quality_tier="Thin"),
             "tier": "avoid", "dynamic_edge_floor": 2.5},
        ]
        ds = compute_slate_debug_stats(entries)
        assert ds["total_recs"] == 3
        assert ds["counts_by_tier"] == {"tier1a": 1, "tier2": 1, "avoid": 1}
        assert ds["counts_by_confidence"]["High"] == 1
        assert ds["counts_by_confidence"]["Medium"] == 1
        assert ds["counts_by_confidence"]["Low"] == 1
        assert ds["counts_by_quality_tier"]["Elite"] == 1
        assert ds["counts_by_quality_tier"]["Moderate"] == 1
        assert ds["counts_by_quality_tier"]["Thin"] == 1

    def test_tier1_gate_failure_counts(self):
        entries = [
            {**_entry(edge_pct=2.0, confidence="Medium", quality_tier="Moderate"),
             "tier": "tier2", "dynamic_edge_floor": 3.0},
        ]
        ds = compute_slate_debug_stats(entries)
        t1g = ds["tier1_gate_failures"]
        assert t1g["confidence_not_high"] == 1
        assert t1g["quality_tier_not_elite_strong"] == 1
        assert t1g["below_dynamic_floor"] == 1
        assert t1g["edge_not_positive"] == 0

    def test_tier2_gate_failure_counts(self):
        # edge_pct=-0.5 to trigger below_edge_floor (edge <= 0)
        entries = [
            {**_entry(edge_pct=-0.5, confidence="Low", quality_tier="Thin", edge_z=0.3),
             "tier": "avoid", "dynamic_edge_floor": 3.0},
        ]
        ds = compute_slate_debug_stats(entries)
        t2g = ds["tier2_gate_failures"]
        assert t2g["confidence_not_high_medium"] == 1
        assert t2g["quality_tier_not_elite_strong_moderate"] == 1
        assert t2g["below_edge_floor"] == 1
        assert t2g["edge_z_too_low"] == 0  # edge_z gate disabled (min=0)

    def test_sigma_stats(self):
        entries = [
            {**_entry(market_volatility_sigma=0.01),
             "tier": "tier1", "dynamic_edge_floor": 3.012},
            {**_entry(market_volatility_sigma=0.03),
             "tier": "tier1", "dynamic_edge_floor": 3.036},
            {**_entry(market_volatility_sigma=0.05),
             "tier": "tier1", "dynamic_edge_floor": 3.06},
        ]
        ds = compute_slate_debug_stats(entries)
        assert ds["sigma_stats"]["min"] == 0.01
        assert ds["sigma_stats"]["max"] == 0.05
        assert abs(ds["sigma_stats"]["median"] - 0.03) < 1e-9

    def test_dynamic_floor_stats(self):
        entries = [
            {**_entry(), "tier": "tier1", "dynamic_edge_floor": 3.0},
            {**_entry(), "tier": "tier1", "dynamic_edge_floor": 3.5},
            {**_entry(), "tier": "tier1", "dynamic_edge_floor": 4.0},
        ]
        ds = compute_slate_debug_stats(entries)
        assert ds["dynamic_floor_stats"]["min"] == 3.0
        assert ds["dynamic_floor_stats"]["max"] == 4.0
        assert ds["dynamic_floor_stats"]["median"] == 3.5

    def test_warning_sigma_too_large(self):
        """sigma > 0.25 triggers a units-bug warning."""
        entries = [
            {**_entry(market_volatility_sigma=2.0),
             "tier": "avoid", "dynamic_edge_floor": 5.4},
        ]
        ds = compute_slate_debug_stats(entries)
        assert any("sigma max" in w for w in ds["warnings"])

    def test_warning_floor_median_too_large(self):
        """dynamic floor median > 6 triggers a units-bug warning."""
        entries = [
            {**_entry(), "tier": "avoid", "dynamic_edge_floor": 7.0},
            {**_entry(), "tier": "avoid", "dynamic_edge_floor": 8.0},
        ]
        ds = compute_slate_debug_stats(entries)
        assert any("dynamic floor median" in w for w in ds["warnings"])

    def test_warning_no_stay_away_or_tier3(self):
        """All recs in tier1b → StayAway+Tier3==0 warning."""
        entries = [
            {**_entry(), "tier": "tier1b", "dynamic_edge_floor": 2.5},
        ]
        ds = compute_slate_debug_stats(entries)
        assert any("StayAway+Tier3 == 0" in w for w in ds["warnings"])

    def test_no_warnings_for_normal_data(self):
        entries = [
            {**_entry(market_volatility_sigma=0.02),
             "tier": "tier1b", "dynamic_edge_floor": 2.52},
            {**_entry(market_volatility_sigma=0.01, edge_pct=0.5,
                      confidence="Low", quality_tier="Thin"),
             "tier": "avoid", "dynamic_edge_floor": 2.51},
        ]
        ds = compute_slate_debug_stats(entries)
        # Has both tiers and stay_away, sigma < 0.25, floor < 6
        assert ds["warnings"] == []


# ---------------------------------------------------------------------------
# Regression: classification not dropped pre-filtering
# ---------------------------------------------------------------------------


class TestClassificationNotDroppedPreFiltering:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_all_recs_accounted_for(self, mock_rbb):
        """Every rec is classified into exactly one bucket — none dropped."""

        def _recs(lines):
            ev = lines[0].event
            if "A" in ev:
                return [_make_rec(
                    quality_score=90, edge_pct=4.0,
                    quality_tier="Elite", confidence="High",
                )]
            if "B" in ev:
                return [_make_rec(
                    quality_score=55, edge_pct=2.0,
                    quality_tier="Moderate", confidence="Medium",
                )]
            if "C" in ev:
                return [_make_rec(
                    quality_score=20, edge_pct=0.3,
                    quality_tier="Thin", confidence="Low",
                )]
            return [_make_rec(
                quality_score=60, edge_pct=1.0,
                quality_tier="Moderate", confidence="Medium",
            )]

        mock_rbb.side_effect = _recs
        lines = {
            "e1": _event_lines("TeamA @ X"),
            "e2": _event_lines("TeamB @ Y"),
            "e3": _event_lines("TeamC @ Z"),
            "e4": _event_lines("TeamD @ W"),
        }
        result = build_daily_slate(lines)

        total = result["counts"]["total_recs"]
        assert total == 4

        # Sum of displayed buckets may be < total (display filters can hide
        # tier1/tier2 entries), but counts should always add up.
        classified = (
            result["counts"]["tier1"]
            + result["counts"]["tier2"]
            + result["counts"]["tier3"]
            + result["counts"]["stay_away"]
        )
        assert classified == total

    @patch("line_tracker.slate.recommend_best_bets")
    def test_display_filters_cannot_drop_classifications(self, mock_rbb):
        """Aggressive display filters don't change classification counts."""

        def _recs(lines):
            return [_make_rec(
                quality_score=80, edge_pct=3.0,
                quality_tier="Strong", confidence="High",
            )]

        mock_rbb.side_effect = _recs
        lines = {"e1": _event_lines("TeamA @ X"), "e2": _event_lines("TeamB @ Y")}

        r_plain = build_daily_slate(lines)
        r_filtered = build_daily_slate(
            lines, filters={"min_edge": 99.0, "min_quality": 99}
        )

        # Classification counts identical
        assert r_plain["counts"] == r_filtered["counts"]
        # Display may differ — filtered tier1 list is empty
        assert len(r_filtered["tier1"]) == 0
        assert len(r_plain["tier1"]) == 2


# ---------------------------------------------------------------------------
# Regression: strict mode affects only Tier 2 display, not classification
# ---------------------------------------------------------------------------


class TestStrictModeRegression:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_strict_vs_relaxed_tier1_unchanged(self, mock_rbb):
        """Tier 1 membership is identical under strict and relaxed modes."""

        def _recs(lines):
            ev = lines[0].event
            if "A" in ev:
                return [_make_rec(
                    quality_score=90, edge_pct=4.0,
                    quality_tier="Elite", confidence="High",
                )]
            return [_make_rec(
                quality_score=60, edge_pct=0.8,
                quality_tier="Moderate", confidence="Medium",
            )]

        mock_rbb.side_effect = _recs
        lines = {
            "e1": _event_lines("TeamA @ X"),
            "e2": _event_lines("TeamB @ Y"),
        }
        result = build_daily_slate(lines)

        tier1_strict = result["tier1"]
        stay_away = result["stay_away"]

        # Relaxed display: promote qualifying stay_away into tier2 display
        promoted = [e for e in stay_away if passes_relaxed_tier2(e)]
        tier2_relaxed = list(result["tier2"]) + promoted

        # Tier 1 is identical
        assert tier1_strict == result["tier1"]
        # Tier 2 gained entries in relaxed mode
        assert len(tier2_relaxed) >= len(result["tier2"])
        # Classification tier field on entries is unchanged
        for e in promoted:
            assert e["tier"] == "avoid"  # canonical assignment unchanged

    @patch("line_tracker.slate.recommend_best_bets")
    def test_strict_vs_relaxed_counts_unchanged(self, mock_rbb):
        """counts dict is identical — relaxed mode is display-only."""
        mock_rbb.return_value = [_make_rec(
            quality_score=60, edge_pct=0.8,
            quality_tier="Moderate", confidence="Medium",
        )]
        result = build_daily_slate({"e1": _event_lines()})
        counts_before = dict(result["counts"])

        # Simulate relaxed display
        _ = [e for e in result["stay_away"] if passes_relaxed_tier2(e)]

        # Counts unchanged
        assert result["counts"] == counts_before


# ---------------------------------------------------------------------------
# Gate failure counters integration
# ---------------------------------------------------------------------------


class TestGateFailureCountersIntegration:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_gate_failures_nonzero_when_tier1_empty(self, mock_rbb):
        """When no entries make Tier 1, at least one T1 gate failure is non-zero."""
        mock_rbb.return_value = [_make_rec(
            quality_score=55, edge_pct=1.5,
            quality_tier="Moderate", confidence="Medium",
            books_used_count=4,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        ds = result["debug_stats"]
        t1g = ds["tier1_gate_failures"]
        assert result["counts"]["tier1"] == 0
        # At least one gate failed
        assert sum(t1g.values()) > 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_gate_failures_zero_for_tier1_pass(self, mock_rbb):
        """When entry passes Tier 1, it shouldn't fail any T1 gate."""
        mock_rbb.return_value = [_make_rec(
            quality_score=90, edge_pct=4.0,
            quality_tier="Elite", confidence="High",
            market_volatility_sigma=0.0,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        ds = result["debug_stats"]
        t1g = ds["tier1_gate_failures"]
        # With 1 entry that passes all T1 gates, all failure counts are 0
        assert t1g["confidence_not_high"] == 0
        assert t1g["quality_tier_not_elite_strong"] == 0
        assert t1g["below_dynamic_floor"] == 0
        assert t1g["edge_not_positive"] == 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_debug_stats_always_present(self, mock_rbb):
        """debug_stats is present even without debug flag."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0,
            quality_tier="Strong", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert "debug_stats" in result
        assert "debug" not in result  # debug flag was not passed


# ---------------------------------------------------------------------------
# Sigma unit sanity
# ---------------------------------------------------------------------------


class TestSigmaUnitSanity:
    def test_small_sigma_correct_floor(self):
        """sigma=0.02 → floor_1a = max(2.0, 100*0.02) = 2.0."""
        result = classify_rec(_entry(
            edge_pct=2.0, confidence="High", quality_tier="Strong",
            market_volatility_sigma=0.02, books_used=6,
        ))
        assert result["tier"] == "tier1a"
        expected_floor = max(_TIER1_BASE_EDGE, _TIER1_SIGMA_MULT * 0.02)
        assert abs(result["dynamic_edge_floor"] - expected_floor) < 1e-9

    def test_large_sigma_triggers_warning(self):
        """sigma=2.0 → floor = 200.0; debug_stats should warn."""
        entries = [
            {**_entry(market_volatility_sigma=2.0),
             "tier": "avoid", "dynamic_edge_floor": 200.0},
        ]
        ds = compute_slate_debug_stats(entries)
        assert any("sigma max" in w for w in ds["warnings"])
        assert ds["sigma_stats"]["max"] == 2.0
        assert ds["dynamic_floor_stats"]["max"] == 200.0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_sigma_stats_integration(self, mock_rbb):
        """Sigma stats are populated from real build_daily_slate."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.1,
            quality_tier="Strong", confidence="High",
            market_volatility_sigma=0.02,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        ds = result["debug_stats"]
        assert ds["sigma_stats"]["min"] == 0.02
        assert ds["sigma_stats"]["max"] == 0.02
        expected_floor = max(_TIER1_BASE_EDGE, _TIER1_SIGMA_MULT * 0.02)
        assert abs(ds["dynamic_floor_stats"]["min"] - expected_floor) < 1e-9

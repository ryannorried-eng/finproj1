"""Tests for enhanced Stay Away avoid flags, composite scoring, and ranking."""

from datetime import datetime
from unittest.mock import patch

from line_tracker.best_bets import BetRecommendation
from line_tracker.models import BettingLine, BetType
from line_tracker.slate import (
    _AVOID_DIVERGENCE_MIN,
    _AVOID_HOLD_MAX,
    _add_market_quality_flags,
    _avoid_score,
    _stay_away_sort_key,
    build_daily_slate,
    classify_rec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 1, 12, 0)


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
    divergence: float | None = None,
    market: str = "moneyline",
) -> dict:
    """Build a minimal entry dict for classify_rec / _avoid_score."""
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
        "divergence": divergence,
        "market": market,
    }


def _ml_line(
    sportsbook: str,
    home_odds: float,
    away_odds: float,
    event: str = "Lakers @ Celtics",
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
    )


def _event_lines(event: str = "Lakers @ Celtics"):
    return [
        _ml_line("FanDuel", -150, 130, event=event),
        _ml_line("DraftKings", -145, 125, event=event),
    ]


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
) -> BetRecommendation:
    return BetRecommendation(
        market=market,
        selection=selection,
        side="home",
        line=None,
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
        market_hold_median=market_hold_median,
        edge_z=edge_z,
    )


# ---------------------------------------------------------------------------
# _add_market_quality_flags
# ---------------------------------------------------------------------------


class TestAddMarketQualityFlags:
    def test_high_hold_flag(self):
        reasons: list[str] = []
        entry = _entry(market_hold_median=8.5)
        _add_market_quality_flags(reasons, entry)
        assert any("High market hold" in r for r in reasons)
        assert any("8.5%" in r for r in reasons)

    def test_hold_at_threshold_no_flag(self):
        reasons: list[str] = []
        entry = _entry(market_hold_median=_AVOID_HOLD_MAX)
        _add_market_quality_flags(reasons, entry)
        assert not any("High market hold" in r for r in reasons)

    def test_noisy_market_flag(self):
        reasons: list[str] = []
        entry = _entry(market_volatility_sigma=0.06, edge_pct=1.5)
        _add_market_quality_flags(reasons, entry)
        assert any("Noisy market" in r for r in reasons)

    def test_noisy_market_needs_both_conditions(self):
        """High sigma alone (with high edge) does not trigger noise flag."""
        reasons: list[str] = []
        entry = _entry(market_volatility_sigma=0.06, edge_pct=3.0)
        _add_market_quality_flags(reasons, entry)
        assert not any("Noisy market" in r for r in reasons)

    def test_noisy_market_low_edge_alone_not_enough(self):
        """Low edge alone (with low sigma) does not trigger noise flag."""
        reasons: list[str] = []
        entry = _entry(market_volatility_sigma=0.03, edge_pct=1.0)
        _add_market_quality_flags(reasons, entry)
        assert not any("Noisy market" in r for r in reasons)

    def test_divergence_flag(self):
        reasons: list[str] = []
        entry = _entry(divergence=0.06)
        _add_market_quality_flags(reasons, entry)
        assert any("Sharp-retail divergence" in r for r in reasons)

    def test_divergence_at_threshold_no_flag(self):
        reasons: list[str] = []
        entry = _entry(divergence=_AVOID_DIVERGENCE_MIN)
        _add_market_quality_flags(reasons, entry)
        assert not any("Sharp-retail divergence" in r for r in reasons)

    def test_divergence_none_no_flag(self):
        reasons: list[str] = []
        entry = _entry(divergence=None)
        _add_market_quality_flags(reasons, entry)
        assert not any("Sharp-retail divergence" in r for r in reasons)

    def test_all_flags_together(self):
        reasons: list[str] = []
        entry = _entry(
            market_hold_median=9.0,
            market_volatility_sigma=0.07,
            edge_pct=1.0,
            divergence=0.05,
        )
        _add_market_quality_flags(reasons, entry)
        assert len(reasons) == 3
        assert any("High market hold" in r for r in reasons)
        assert any("Noisy market" in r for r in reasons)
        assert any("Sharp-retail divergence" in r for r in reasons)

    def test_no_flags_clean_market(self):
        reasons: list[str] = []
        entry = _entry(
            market_hold_median=3.0,
            market_volatility_sigma=0.01,
            edge_pct=4.0,
            divergence=0.01,
        )
        _add_market_quality_flags(reasons, entry)
        assert reasons == []


# ---------------------------------------------------------------------------
# _avoid_score
# ---------------------------------------------------------------------------


class TestAvoidScore:
    def test_clean_entry_zero_score(self):
        entry = _entry(confidence="High", quality_tier="Strong")
        assert _avoid_score(entry) == 0.0

    def test_low_confidence_adds_30(self):
        score = _avoid_score(_entry(confidence="Low"))
        assert score >= 30.0

    def test_unknown_confidence_adds_20(self):
        score = _avoid_score(_entry(confidence="Unknown"))
        assert score >= 20.0

    def test_thin_quality_adds_25(self):
        score = _avoid_score(_entry(quality_tier="Thin"))
        assert score >= 25.0

    def test_unknown_quality_adds_15(self):
        score = _avoid_score(_entry(quality_tier="Weak"))
        assert score >= 15.0

    def test_low_edge_z_adds_proportional(self):
        # edge_z=0.5, below 1.0 → 20*(1-0.5) = 10
        score = _avoid_score(_entry(edge_z=0.5))
        assert score >= 10.0

    def test_edge_z_zero_no_penalty(self):
        """edge_z == 0 (unavailable) does not penalize."""
        score = _avoid_score(_entry(edge_z=0.0))
        assert score == 0.0

    def test_high_hold_adds_15(self):
        score = _avoid_score(_entry(market_hold_median=8.0))
        assert score >= 15.0

    def test_noisy_market_adds_20(self):
        score = _avoid_score(
            _entry(market_volatility_sigma=0.06, edge_pct=1.0)
        )
        assert score >= 20.0

    def test_divergence_adds_15(self):
        score = _avoid_score(_entry(divergence=0.05))
        assert score >= 15.0

    def test_cumulative_all_bad(self):
        """Entry with all bad factors should have a high composite score."""
        score = _avoid_score(_entry(
            confidence="Low",
            quality_tier="Thin",
            edge_z=0.5,
            market_hold_median=9.0,
            market_volatility_sigma=0.08,
            edge_pct=1.0,
            divergence=0.06,
        ))
        # 30 (Low) + 25 (Thin) + 10 (edge_z) + 15 (hold) + 20 (noise) + 15 (div)
        assert score >= 115.0

    def test_medium_confidence_no_penalty(self):
        score = _avoid_score(_entry(confidence="Medium"))
        assert score == 0.0


# ---------------------------------------------------------------------------
# classify_rec: market-quality flags in avoid reasons
# ---------------------------------------------------------------------------


class TestClassifyRecMarketFlags:
    def test_avoid_includes_high_hold_reason(self):
        """Stay Away entry with high hold gets the flag reason."""
        e = _entry(
            edge_pct=0.5, confidence="Medium", quality_tier="Moderate",
            market_hold_median=8.0,
        )
        result = classify_rec(e)
        assert result["tier"] == "avoid"
        assert any("High market hold" in r for r in result["reasons"])

    def test_avoid_includes_noise_reason(self):
        """Stay Away entry with high sigma + low edge gets noise flag."""
        e = _entry(
            edge_pct=1.0, confidence="Medium", quality_tier="Moderate",
            market_volatility_sigma=0.06,
        )
        result = classify_rec(e)
        assert result["tier"] == "avoid"
        assert any("Noisy market" in r for r in result["reasons"])

    def test_avoid_includes_divergence_reason(self):
        """Stay Away entry with high divergence gets the flag."""
        e = _entry(
            edge_pct=0.5, confidence="Medium", quality_tier="Moderate",
            divergence=0.06,
        )
        result = classify_rec(e)
        assert result["tier"] == "avoid"
        assert any("Sharp-retail divergence" in r for r in result["reasons"])

    def test_hard_disqualifier_also_gets_market_flags(self):
        """Hard-disqualified entries also get market quality flags."""
        e = _entry(
            market_unstable=True,
            market_hold_median=9.0,
        )
        result = classify_rec(e)
        assert result["tier"] == "avoid"
        assert any("Unstable market" in r for r in result["reasons"])
        assert any("High market hold" in r for r in result["reasons"])

    def test_tier1_unaffected_by_market_flags(self):
        """Tier 1 classification is not affected by market quality flags."""
        e = _entry(
            edge_pct=3.5, confidence="High", quality_tier="Strong",
            market_hold_median=9.0, divergence=0.06,
        )
        result = classify_rec(e)
        assert result["tier"] == "tier1"
        assert result["reasons"] == []

    def test_tier2_unaffected_by_market_flags(self):
        """Tier 2 classification is not affected by market quality flags."""
        e = _entry(
            edge_pct=2.0, confidence="Medium", quality_tier="Moderate",
            market_hold_median=9.0, divergence=0.06,
        )
        result = classify_rec(e)
        assert result["tier"] == "tier2"
        assert result["reasons"] == []

    def test_multiple_market_flags(self):
        """Multiple market flags appear when all conditions met."""
        e = _entry(
            edge_pct=0.5, confidence="Low", quality_tier="Thin",
            market_hold_median=8.0,
            market_volatility_sigma=0.07,
            divergence=0.05,
        )
        result = classify_rec(e)
        assert result["tier"] == "avoid"
        flags = result["reasons"]
        assert any("Confidence Low" in r for r in flags)
        assert any("Quality tier Thin" in r for r in flags)
        assert any("High market hold" in r for r in flags)
        assert any("Noisy market" in r for r in flags)
        assert any("Sharp-retail divergence" in r for r in flags)


# ---------------------------------------------------------------------------
# _stay_away_sort_key: ranking determinism
# ---------------------------------------------------------------------------


class TestStayAwaySortKeyEnhanced:
    def test_higher_avoid_score_sorts_first(self):
        """Entry with higher avoid_score sorts before lower score."""
        bad = _entry(confidence="Low", quality_tier="Thin")
        ok = _entry(confidence="Medium", quality_tier="Moderate",
                    edge_pct=1.0)
        assert _stay_away_sort_key(bad) < _stay_away_sort_key(ok)

    def test_low_confidence_sorts_first(self):
        """Backward compat: low confidence still sorts first."""
        low = _entry(confidence="Low", quality_score=50)
        high = _entry(confidence="High", quality_score=50)
        assert _stay_away_sort_key(low) < _stay_away_sort_key(high)

    def test_lower_edge_z_sorts_first(self):
        """Backward compat: lower edge_z still sorts first."""
        bad = _entry(edge_z=-1.0, quality_score=50)
        ok = _entry(edge_z=0.5, quality_score=50)
        assert _stay_away_sort_key(bad) < _stay_away_sort_key(ok)

    def test_higher_sigma_sorts_first(self):
        """Backward compat: higher sigma still sorts first when tied."""
        vol = _entry(market_volatility_sigma=2.0, quality_score=50)
        calm = _entry(market_volatility_sigma=0.01, quality_score=50)
        assert _stay_away_sort_key(vol) < _stay_away_sort_key(calm)

    def test_high_hold_sorts_before_clean_market(self):
        """Entry with high hold gets higher avoid_score → sorts first."""
        bad_hold = _entry(
            confidence="Low", market_hold_median=9.0,
        )
        clean = _entry(confidence="Low", market_hold_median=3.0)
        assert _stay_away_sort_key(bad_hold) < _stay_away_sort_key(clean)

    def test_divergence_breaks_tie(self):
        """High divergence adds to avoid_score, breaking a tie."""
        with_div = _entry(confidence="Low", divergence=0.06)
        no_div = _entry(confidence="Low", divergence=None)
        assert _stay_away_sort_key(with_div) < _stay_away_sort_key(no_div)

    def test_deterministic_ordering(self):
        """Sort is deterministic across multiple entries."""
        entries = [
            _entry(confidence="High", quality_tier="Moderate", edge_pct=1.0,
                   quality_score=60),
            _entry(confidence="Low", quality_tier="Thin", edge_pct=0.5,
                   edge_z=0.3, market_hold_median=9.0),
            _entry(confidence="Medium", quality_tier="Moderate", edge_pct=1.0,
                   market_volatility_sigma=0.08, divergence=0.06),
        ]
        sorted_entries = sorted(entries, key=_stay_away_sort_key)
        # Worst entry: Low + Thin + low edge_z + high hold → highest score
        assert sorted_entries[0]["confidence"] == "Low"
        # Next: Medium + noise + divergence
        assert sorted_entries[1]["confidence"] == "Medium"
        # Last: High with least bad factors
        assert sorted_entries[2]["confidence"] == "High"


# ---------------------------------------------------------------------------
# Integration: build_daily_slate avoid_score and market fields
# ---------------------------------------------------------------------------


class TestBuildSlateAvoidScore:
    @patch("line_tracker.slate.recommend_best_bets")
    def test_avoid_entry_has_avoid_score(self, mock_rbb):
        """Stay Away entries carry an avoid_score field."""
        mock_rbb.return_value = [_make_rec(
            quality_score=20, edge_pct=0.5,
            quality_tier="Thin", confidence="Low",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["stay_away"]) == 1
        entry = result["stay_away"][0]
        assert "avoid_score" in entry
        assert entry["avoid_score"] > 0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_tier1_has_avoid_score_zero(self, mock_rbb):
        """Tier 1 entries have avoid_score of 0."""
        mock_rbb.return_value = [_make_rec(
            quality_score=90, edge_pct=4.0,
            quality_tier="Elite", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        assert len(result["tier1"]) == 1
        assert result["tier1"][0]["avoid_score"] == 0.0

    @patch("line_tracker.slate.recommend_best_bets")
    def test_entry_has_market_hold_median(self, mock_rbb):
        """Entries carry market_hold_median from the recommendation."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0,
            quality_tier="Strong", confidence="High",
            market_hold_median=4.5,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["tier1"][0]
        assert entry["market_hold_median"] == 4.5

    @patch("line_tracker.slate.recommend_best_bets")
    def test_entry_has_divergence_field(self, mock_rbb):
        """Entries carry a divergence field (may be None)."""
        mock_rbb.return_value = [_make_rec(
            quality_score=80, edge_pct=3.0,
            quality_tier="Strong", confidence="High",
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["tier1"][0]
        assert "divergence" in entry

    @patch("line_tracker.slate.recommend_best_bets")
    def test_stay_away_ranked_by_avoid_score(self, mock_rbb):
        """Stay Away entries are sorted worst-first by avoid_score."""

        def side_effect(lines):
            ev = lines[0].event
            if "Bad" in ev:
                return [_make_rec(
                    quality_score=20, edge_pct=0.3,
                    quality_tier="Thin", confidence="Low",
                    market_hold_median=9.0,
                )]
            return [_make_rec(
                quality_score=50, edge_pct=1.0,
                quality_tier="Moderate", confidence="Medium",
            )]

        mock_rbb.side_effect = side_effect
        lines = {
            "evt1": _event_lines("OK Team @ Others"),
            "evt2": _event_lines("Bad Team @ Worst"),
        }
        result = build_daily_slate(lines)
        sa = result["stay_away"]
        assert len(sa) == 2
        # Worst entry (Bad Team) should be first
        assert sa[0]["avoid_score"] >= sa[1]["avoid_score"]
        assert sa[0]["event"] == "Bad Team @ Worst"

    @patch("line_tracker.slate.recommend_best_bets")
    def test_high_hold_adds_reason_in_slate(self, mock_rbb):
        """An avoid entry with high hold shows the market hold reason."""
        mock_rbb.return_value = [_make_rec(
            quality_score=20, edge_pct=0.5,
            quality_tier="Thin", confidence="Low",
            market_hold_median=8.5,
        )]
        result = build_daily_slate({"evt1": _event_lines()})
        entry = result["stay_away"][0]
        assert any("High market hold" in r for r in entry["avoid_reasons"])

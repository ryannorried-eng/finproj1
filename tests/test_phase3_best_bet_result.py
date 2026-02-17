"""Phase 3 tests: BestBetResult, explanations, edge consistency, rolling cal."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from line_tracker.best_bets import (
    BetRecommendation,
    extract_best_bet_results,
    recommend_best_bet_results,
    recommend_best_bets,
)
from line_tracker.calibration import (
    _DEFAULTS,
    _rolling_score_adjustment,
    calibrate_thresholds,
    compute_rolling_clv_stats,
    grid_search_thresholds,
)
from line_tracker.models import BestBetResult, BettingLine, BetType
from line_tracker.services.explanation_service import (
    format_explanation_summary,
    get_recommendation_explanations,
)

# ── Helpers ──────────────────────────────────────────────────────────────


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
    spread: float,
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
        home_value=spread,
        away_value=-spread,
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


def _sample_lines() -> list[BettingLine]:
    """Build a set of moneyline lines from 6 sportsbooks."""
    return [
        _ml_line("Pinnacle", -150, 130),
        _ml_line("DraftKings", -145, 125),
        _ml_line("FanDuel", -155, 135),
        _ml_line("BetMGM", -148, 128),
        _ml_line("Caesars", -150, 130),
        _ml_line("BetOnline", -152, 132),
    ]


def _sample_spread_lines() -> list[BettingLine]:
    """Build a set of spread lines from 6 sportsbooks."""
    return [
        _spread_line("Pinnacle", -3.5, -110, -110),
        _spread_line("DraftKings", -3.5, -108, -112),
        _spread_line("FanDuel", -3.5, -110, -110),
        _spread_line("BetMGM", -3.5, -112, -108),
        _spread_line("Caesars", -3.5, -110, -110),
        _spread_line("BetOnline", -3.5, -110, -110),
    ]


def _sample_total_lines() -> list[BettingLine]:
    """Build a set of total lines from 6 sportsbooks."""
    return [
        _total_line("Pinnacle", 215.5, -110, -110),
        _total_line("DraftKings", 215.5, -108, -112),
        _total_line("FanDuel", 215.5, -110, -110),
        _total_line("BetMGM", 215.5, -112, -108),
        _total_line("Caesars", 215.5, -110, -110),
        _total_line("BetOnline", 215.5, -110, -110),
    ]


def _make_training_df(
    n: int = 300,
    *,
    edge_ev_100: float = 2.0,
    edge_z: float = 2.0,
    hold: float = 5.0,
    books_used: int = 6,
    beat_rate: float = 0.60,
    avg_clv: float = 0.003,
    days: int = 90,
) -> pd.DataFrame:
    """Create a synthetic training DataFrame for calibration tests."""
    import random

    random.seed(42)
    start = date.today() - timedelta(days=days)
    rows = []
    for i in range(n):
        beat = 1 if random.random() < beat_rate else 0
        if beat:
            clv = avg_clv + random.gauss(0, 0.001)
        else:
            clv = -abs(avg_clv) + random.gauss(0, 0.001)
        rows.append({
            "edge_ev_100": edge_ev_100,
            "edge_z": edge_z,
            "hold": hold,
            "books_used": books_used,
            "beat": beat,
            "clv": clv,
            "date": start + timedelta(days=i % days),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 1. BestBetResult domain model
# ═══════════════════════════════════════════════════════════════════════


class TestBestBetResultModel:
    """BestBetResult domain model field presence and types."""

    def test_required_fields_present(self):
        bbr = BestBetResult(
            edge_pct=2.5,
            consensus_prob=0.55,
            best_odds_american=-150,
            best_odds_decimal=1.6667,
            books_used=["Pinnacle", "DraftKings"],
            volatility_sigma=0.02,
            recency_weight=0.95,
            outliers_removed=1,
        )
        assert bbr.edge_pct == 2.5
        assert bbr.consensus_prob == 0.55
        assert bbr.best_odds_american == -150
        assert bbr.best_odds_decimal == 1.6667
        assert bbr.books_used == ["Pinnacle", "DraftKings"]
        assert bbr.volatility_sigma == 0.02
        assert bbr.recency_weight == 0.95
        assert bbr.outliers_removed == 1
        assert bbr.explanation == {}
        assert bbr.raw_inputs is None

    def test_explanation_dict(self):
        bbr = BestBetResult(
            edge_pct=3.0,
            consensus_prob=0.55,
            best_odds_american=130,
            best_odds_decimal=2.30,
            books_used=["A", "B"],
            volatility_sigma=0.02,
            recency_weight=0.90,
            outliers_removed=0,
            explanation={"consensus_method": "trimmed_mean", "key": "value"},
        )
        assert bbr.explanation["consensus_method"] == "trimmed_mean"
        assert "key" in bbr.explanation

    def test_raw_inputs_optional(self):
        bbr = BestBetResult(
            edge_pct=1.0,
            consensus_prob=0.52,
            best_odds_american=-110,
            best_odds_decimal=1.909,
            books_used=[],
            volatility_sigma=0.0,
            recency_weight=1.0,
            outliers_removed=0,
            raw_inputs={"debug_data": [1, 2, 3]},
        )
        assert bbr.raw_inputs is not None
        assert bbr.raw_inputs["debug_data"] == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════
# 2. Ranking explanation fields present
# ═══════════════════════════════════════════════════════════════════════


class TestExplanationFieldsPresent:
    """Every BetRecommendation carries a BestBetResult with full explanation."""

    def test_moneyline_has_best_bet_result(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            assert hasattr(rec, "best_bet_result"), (
                f"Rec {rec.market}/{rec.side} missing best_bet_result"
            )
            bbr = rec.best_bet_result
            assert isinstance(bbr, BestBetResult)

    def test_explanation_has_required_sections(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            expl = bbr.explanation
            assert "consensus_method" in expl
            assert "edge_breakdown" in expl
            assert "ev_edge" in expl
            assert "confidence_reasoning" in expl
            assert "quality_factors" in expl
            assert "outlier_info" in expl
            assert "market_context" in expl
            assert "recency" in expl
            assert "kelly" in expl

    def test_explanation_edge_breakdown_complete(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=2, now=datetime(2025, 1, 1, 12, 5))
        rec = recs[0]
        eb = rec.best_bet_result.explanation["edge_breakdown"]
        assert "consensus_prob" in eb
        assert "breakeven_prob" in eb
        assert "edge_pp" in eb
        assert "edge_pct" in eb
        assert "ev_roi" in eb
        assert "ev_100" in eb

    def test_explanation_quality_factors_complete(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=2, now=datetime(2025, 1, 1, 12, 5))
        rec = recs[0]
        qf = rec.best_bet_result.explanation["quality_factors"]
        assert "edge_score" in qf
        assert "agreement_score" in qf
        assert "coverage_score" in qf
        assert "freshness_score" in qf
        assert "weights" in qf
        assert "quality_score" in qf
        assert "quality_tier" in qf

    def test_explanation_confidence_reasoning(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=2, now=datetime(2025, 1, 1, 12, 5))
        rec = recs[0]
        cr = rec.best_bet_result.explanation["confidence_reasoning"]
        assert "edge_z" in cr
        assert "thresholds" in cr
        assert "result" in cr
        assert cr["result"] in ("High", "Medium", "Low")

    def test_books_used_is_list_of_strings(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=2, now=datetime(2025, 1, 1, 12, 5))
        rec = recs[0]
        bbr = rec.best_bet_result
        assert isinstance(bbr.books_used, list)
        assert len(bbr.books_used) > 0
        assert all(isinstance(b, str) for b in bbr.books_used)

    def test_recency_weight_positive(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=2, now=datetime(2025, 1, 1, 12, 5))
        rec = recs[0]
        assert rec.best_bet_result.recency_weight > 0

    def test_volatility_sigma_non_negative(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=2, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            assert rec.best_bet_result.volatility_sigma >= 0

    def test_spread_explanation(self):
        lines = _sample_spread_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.market == "spread"
            assert "edge_breakdown" in bbr.explanation

    def test_total_explanation(self):
        lines = _sample_total_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.market == "total"
            assert "edge_breakdown" in bbr.explanation


# ═══════════════════════════════════════════════════════════════════════
# 3. Edge calculations consistent with previous behavior
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeConsistency:
    """BestBetResult edge metrics must match BetRecommendation fields."""

    def test_edge_pct_matches_recommendation(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.edge_pct == rec.edge_pct

    def test_consensus_prob_matches(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.consensus_prob == rec.consensus_prob

    def test_best_odds_american_matches(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.best_odds_american == rec.best_odds

    def test_best_odds_decimal_positive(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.best_odds_decimal > 1.0

    def test_ev_100_in_explanation_matches_rec(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=2, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            eb = bbr.explanation["edge_breakdown"]
            assert eb["ev_100"] == rec.ev_100

    def test_edge_z_matches(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.edge_z == rec.edge_z

    def test_quality_score_matches(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.quality_score == rec.quality_score

    def test_quality_tier_matches(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.quality_tier == rec.quality_tier

    def test_spread_edge_consistent(self):
        lines = _sample_spread_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            assert bbr.edge_pct == rec.edge_pct
            assert bbr.consensus_prob == rec.consensus_prob

    def test_outliers_removed_consistent(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            bbr = rec.best_bet_result
            expected = max(0, rec.total_books_count - rec.books_used_count)
            assert bbr.outliers_removed == expected


# ═══════════════════════════════════════════════════════════════════════
# 4. extract_best_bet_results and recommend_best_bet_results
# ═══════════════════════════════════════════════════════════════════════


class TestExtractAndRecommend:
    """Test the convenience functions for extracting BestBetResult."""

    def test_extract_returns_list_of_bbr(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=3, now=datetime(2025, 1, 1, 12, 5))
        results = extract_best_bet_results(recs)
        assert len(results) > 0
        assert all(isinstance(r, BestBetResult) for r in results)

    def test_extract_skips_missing_bbr(self):
        rec_no_bbr = BetRecommendation(
            market="moneyline",
            selection="",
            side="",
            line=None,
            consensus_prob=0.0,
            best_sportsbook="",
            best_odds=0.0,
            breakeven_prob=0.0,
            ev=0.0,
            edge_pct=0.0,
            ev_per_100=0.0,
            confidence="Low",
            skipped_reason="3-way market",
        )
        results = extract_best_bet_results([rec_no_bbr])
        assert results == []

    def test_recommend_best_bet_results_returns_bbr(self):
        lines = _sample_lines()
        results = recommend_best_bet_results(
            lines, top_n=3, now=datetime(2025, 1, 1, 12, 5),
        )
        assert len(results) > 0
        assert all(isinstance(r, BestBetResult) for r in results)

    def test_recommend_best_bet_results_with_raw_inputs(self):
        lines = _sample_lines()
        results = recommend_best_bet_results(
            lines, top_n=2, now=datetime(2025, 1, 1, 12, 5),
            include_raw_inputs=True,
        )
        for r in results:
            assert r.raw_inputs is not None


# ═══════════════════════════════════════════════════════════════════════
# 5. Explanation service
# ═══════════════════════════════════════════════════════════════════════


class TestExplanationService:
    """Explanation service surfaces structured data for Debug mode."""

    def test_get_recommendation_explanations(self):
        lines = _sample_lines()
        explanations = get_recommendation_explanations(lines, top_n=3)
        assert len(explanations) > 0
        for expl in explanations:
            assert "market" in expl
            assert "edge_pct" in expl
            assert "explanation" in expl
            assert isinstance(expl["explanation"], dict)

    def test_explanation_has_all_debug_fields(self):
        lines = _sample_lines()
        explanations = get_recommendation_explanations(lines, top_n=1)
        e = explanations[0]
        required_keys = {
            "market", "selection", "side", "line",
            "edge_pct", "consensus_prob",
            "best_odds_american", "best_odds_decimal",
            "best_sportsbook", "confidence",
            "quality_score", "quality_tier",
            "books_used", "books_used_count",
            "volatility_sigma", "recency_weight",
            "outliers_removed", "edge_z", "edge_ev",
            "edge_ev_shrunk", "ev_roi", "ev_100",
            "kelly_suggested", "sizing_note",
            "explanation",
        }
        missing = required_keys - set(e.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_format_explanation_summary(self):
        lines = _sample_lines()
        recs = recommend_best_bets(lines, top_n=1, now=datetime(2025, 1, 1, 12, 5))
        results = extract_best_bet_results(recs)
        summary = format_explanation_summary(results[0])
        assert isinstance(summary, str)
        assert "MONEYLINE" in summary or "Edge:" in summary
        assert "Quality" in summary


# ═══════════════════════════════════════════════════════════════════════
# 6. Rolling calibration logic
# ═══════════════════════════════════════════════════════════════════════


class TestRollingCalibration:
    """Rolling CLV stats affect auto threshold calibration."""

    def test_compute_rolling_clv_stats_empty_df(self):
        df = pd.DataFrame()
        stats = compute_rolling_clv_stats(df)
        assert stats["has_rolling_data"] is False
        assert stats["rolling_clv_mean"] == 0.0
        assert stats["rolling_clv_std"] == 0.0

    def test_compute_rolling_clv_stats_with_data(self):
        df = _make_training_df(n=100, days=60)
        stats = compute_rolling_clv_stats(df, window_days=30)
        assert stats["has_rolling_data"] is True
        assert stats["rolling_n"] > 0
        assert isinstance(stats["rolling_clv_mean"], float)
        assert isinstance(stats["rolling_clv_std"], float)
        assert stats["window_days"] == 30

    def test_compute_rolling_clv_stats_insufficient_data(self):
        df = _make_training_df(n=5, days=5)
        stats = compute_rolling_clv_stats(df, window_days=30)
        assert stats["has_rolling_data"] is False

    def test_rolling_score_adjustment_no_data(self):
        adj = _rolling_score_adjustment({})
        assert adj == 0.0

    def test_rolling_score_adjustment_positive_clv(self):
        stats = {
            "has_rolling_data": True,
            "rolling_clv_mean": 0.005,
            "rolling_clv_std": 0.002,
        }
        adj = _rolling_score_adjustment(stats)
        # Positive CLV mean should contribute positively
        mean_part = 300.0 * 0.005  # 1.5
        vol_part = 200.0 * 0.002   # 0.4
        expected = mean_part - vol_part
        assert abs(adj - expected) < 0.001

    def test_rolling_score_adjustment_negative_clv(self):
        stats = {
            "has_rolling_data": True,
            "rolling_clv_mean": -0.003,
            "rolling_clv_std": 0.005,
        }
        adj = _rolling_score_adjustment(stats)
        assert adj < 0  # Negative CLV + high vol should give negative adjustment

    def test_calibrate_thresholds_includes_rolling_clv(self):
        df = _make_training_df(n=300, days=90)
        result = calibrate_thresholds(df)
        assert "rolling_clv" in result
        rolling = result["rolling_clv"]
        assert "rolling_clv_mean" in rolling
        assert "rolling_clv_std" in rolling

    def test_calibrate_thresholds_rolling_disabled(self):
        df = _make_training_df(n=300, days=90)
        result = calibrate_thresholds(df, use_rolling=False)
        assert "rolling_clv" in result
        # When disabled, rolling_clv should be empty dict
        assert result["rolling_clv"] == {}

    def test_auto_threshold_adjusts_with_synthetic_clv(self):
        """Auto threshold should differ when rolling CLV trends change.

        Create two datasets: one with positive recent CLV, one with
        negative recent CLV. The negative-CLV dataset should produce
        different (likely stricter) thresholds.
        """
        import random
        random.seed(99)

        # Dataset 1: good recent CLV
        start = date.today() - timedelta(days=90)
        rows_good = []
        for i in range(300):
            beat = 1 if random.random() < 0.60 else 0
            clv = (0.004 + random.gauss(0, 0.001) if beat
                   else -0.002 + random.gauss(0, 0.001))
            rows_good.append({
                "edge_ev_100": 1.5,
                "edge_z": 1.5,
                "hold": 5.0,
                "books_used": 6,
                "beat": beat,
                "clv": clv,
                "date": start + timedelta(days=i % 90),
            })
        df_good = pd.DataFrame(rows_good)

        # Dataset 2: bad recent CLV (same structure but CLV is negative)
        random.seed(99)
        rows_bad = []
        for i in range(300):
            beat = 1 if random.random() < 0.45 else 0
            clv = (-0.003 + random.gauss(0, 0.003) if beat
                   else -0.005 + random.gauss(0, 0.003))
            rows_bad.append({
                "edge_ev_100": 1.5,
                "edge_z": 1.5,
                "hold": 5.0,
                "books_used": 6,
                "beat": beat,
                "clv": clv,
                "date": start + timedelta(days=i % 90),
            })
        df_bad = pd.DataFrame(rows_bad)

        # The rolling stats should differ
        stats_good = compute_rolling_clv_stats(df_good)
        stats_bad = compute_rolling_clv_stats(df_bad)

        if stats_good.get("has_rolling_data") and stats_bad.get("has_rolling_data"):
            adj_good = _rolling_score_adjustment(stats_good)
            adj_bad = _rolling_score_adjustment(stats_bad)
            # Good CLV should produce a more positive adjustment than bad CLV
            assert adj_good > adj_bad

    def test_grid_search_with_rolling_stats(self):
        # Use data with varied values that can pass grid search filters
        # tier2 needs n_min=200, legs_day 5-25
        import random
        random.seed(77)
        start = date.today() - timedelta(days=90)
        rows = []
        for i in range(500):
            ev = random.choice([0.25, 0.5, 0.75, 1.0])
            ez = random.choice([0.5, 1.0, 1.5])
            rows.append({
                "edge_ev_100": ev,
                "edge_z": ez,
                "hold": random.choice([5.0, 6.0, 7.0]),
                "books_used": random.choice([4, 5, 6]),
                "beat": 1 if random.random() < 0.55 else 0,
                "clv": 0.002 + random.gauss(0, 0.002),
                "date": start + timedelta(days=i % 90),
            })
        df = pd.DataFrame(rows)
        rolling = compute_rolling_clv_stats(df)

        # Run with and without rolling stats
        result_with = grid_search_thresholds(df, "tier2", rolling_stats=rolling)
        result_without = grid_search_thresholds(df, "tier2", rolling_stats=None)

        # Both should produce valid results
        assert not result_with.get("fallback_used", True)
        assert not result_without.get("fallback_used", True)

        # With rolling stats, rolling metadata may be attached
        if rolling.get("has_rolling_data"):
            assert "rolling_clv_mean" in result_with

    def test_calibrate_empty_df_with_rolling(self):
        result = calibrate_thresholds(pd.DataFrame())
        assert result["fallback_used"] is True
        assert "rolling_clv" in result
        assert result["rolling_clv"]["has_rolling_data"] is False

    def test_rolling_stats_window_respected(self):
        """Only data within the window should be included."""
        import random
        random.seed(42)

        start = date.today() - timedelta(days=100)
        rows = []
        for i in range(100):
            d = start + timedelta(days=i)
            # Old data has negative CLV, recent data has positive CLV
            if i < 60:
                clv = -0.005
            else:
                clv = 0.005
            rows.append({
                "edge_ev_100": 2.0,
                "edge_z": 2.0,
                "hold": 5.0,
                "books_used": 6,
                "beat": 1 if clv > 0 else 0,
                "clv": clv,
                "date": d,
            })
        df = pd.DataFrame(rows)

        stats_30 = compute_rolling_clv_stats(df, window_days=30)
        stats_90 = compute_rolling_clv_stats(df, window_days=90)

        # 30-day window should capture mostly positive CLV
        # 90-day window should mix positive and negative
        if stats_30.get("has_rolling_data") and stats_90.get("has_rolling_data"):
            assert stats_30["rolling_clv_mean"] > stats_90["rolling_clv_mean"]


# ═══════════════════════════════════════════════════════════════════════
# 7. Calibration backward compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestCalibrationBackwardCompat:
    """Existing calibration behavior is preserved."""

    def test_grid_search_without_rolling_still_works(self):
        df = _make_training_df(n=300, days=90)
        result = grid_search_thresholds(df, "tier1a")
        assert "edge_ev_100" in result
        assert "edge_z" in result
        assert "hold_max" in result
        assert "books_min" in result

    def test_calibrate_thresholds_backward_compat(self):
        df = _make_training_df(n=300, days=90)
        result = calibrate_thresholds(df)
        # All existing keys should still be present
        assert "tier1a" in result
        assert "tier1b" in result
        assert "tier2" in result
        assert "training_rows" in result
        assert "date_range" in result
        assert "fallback_used" in result

    def test_fallback_defaults_unchanged(self):
        assert _DEFAULTS["tier1a"]["edge_ev_100"] == 2.0
        assert _DEFAULTS["tier1b"]["edge_ev_100"] == 1.0
        assert _DEFAULTS["tier2"]["edge_ev_100"] == 0.5

    def test_json_serialization_with_rolling(self):
        from line_tracker.calibration import calibration_from_json, calibration_to_json

        df = _make_training_df(n=300, days=90)
        result = calibrate_thresholds(df)
        json_str = calibration_to_json(result)
        restored = calibration_from_json(json_str)
        assert "tier1a" in restored
        assert "rolling_clv" in restored


# ═══════════════════════════════════════════════════════════════════════
# 8. Mixed market types
# ═══════════════════════════════════════════════════════════════════════


class TestMixedMarkets:
    """BestBetResult works across ML, spread, and total markets together."""

    def test_mixed_lines_all_have_bbr(self):
        lines = _sample_lines() + _sample_spread_lines() + _sample_total_lines()
        recs = recommend_best_bets(lines, top_n=6, now=datetime(2025, 1, 1, 12, 5))
        for rec in recs:
            if rec.skipped_reason:
                continue
            assert hasattr(rec, "best_bet_result")
            assert isinstance(rec.best_bet_result, BestBetResult)

    def test_recommend_best_bet_results_mixed(self):
        lines = _sample_lines() + _sample_spread_lines() + _sample_total_lines()
        results = recommend_best_bet_results(
            lines, top_n=6, now=datetime(2025, 1, 1, 12, 5),
        )
        markets = {r.market for r in results}
        # Should have at least moneyline results
        assert "moneyline" in markets

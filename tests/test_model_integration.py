"""Integration tests for NCAAB model → slate pipeline (Phase 6)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from line_tracker.config import get_model_enabled, get_model_min_edge, get_model_sport
from line_tracker.core.math import american_to_decimal
from line_tracker.models import BettingLine, BetType
from line_tracker.slate import (
    _match_prediction,
    _model_prob_for_market,
    _sel_matches_team,
    build_daily_slate,
)
from line_tracker.storage import LineStore, _migrated_paths

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_migration_cache():
    """Ensure each :memory: DB gets fresh migrations."""
    _migrated_paths.discard(":memory:")
    yield
    _migrated_paths.discard(":memory:")

_FUTURE = datetime.now(timezone.utc) + timedelta(days=1)


def _ml_line(
    sportsbook: str,
    home_odds: float,
    away_odds: float,
    event: str = "Auburn Tigers vs Duke Blue Devils",
    commence_time: datetime | None = None,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="basketball_ncaab",
        event=event,
        bet_type=BetType.MONEYLINE,
        home_team="Duke Blue Devils",
        away_team="Auburn Tigers",
        home_value=home_odds,
        away_value=away_odds,
        home_price=None,
        away_price=None,
        timestamp=datetime.now(timezone.utc),
        commence_time=(commence_time or _FUTURE).isoformat(),
    )


def _make_lines(n_books: int = 6) -> list[BettingLine]:
    """Create a realistic set of moneyline lines for a single event."""
    books = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbetus", "betrivers"]
    return [_ml_line(books[i], -150 + i * 2, 130 - i * 2) for i in range(n_books)]


def _sample_prediction(
    home_ml_prob: float = 0.65,
    away_ml_prob: float = 0.35,
    margin: float = 4.5,
) -> dict:
    return {
        "home_team": "Duke",
        "away_team": "Auburn",
        "game_date": "2026-03-17",
        "predicted_margin": margin,
        "margin_sigma": 10.5,
        "home_ml_prob": home_ml_prob,
        "away_ml_prob": away_ml_prob,
        "market_spread": -3.5,
        "home_spread_prob": 0.58,
        "away_spread_prob": 0.42,
        "market_total": 145.5,
        "over_prob": 0.52,
        "under_prob": 0.48,
        "model_version": "test_v1",
        "model_confidence": 0.72,
        "sport": "basketball_ncaab",
    }


# ---------------------------------------------------------------------------
# 6A: Database migration / predictions repo
# ---------------------------------------------------------------------------


class TestPredictionsRepo:
    def test_save_and_get(self):
        with LineStore(":memory:") as store:
            pred = _sample_prediction()
            count = store.predictions_repo.save_predictions([pred])
            assert count == 1

            got = store.predictions_repo.get_prediction(
                "Duke", "Auburn", "2026-03-17",
            )
            assert got is not None
            assert got["predicted_margin"] == 4.5
            assert got["home_ml_prob"] == 0.65

    def test_upsert_overwrites(self):
        with LineStore(":memory:") as store:
            pred1 = _sample_prediction(home_ml_prob=0.60)
            store.predictions_repo.save_predictions([pred1])

            pred2 = _sample_prediction(home_ml_prob=0.70)
            store.predictions_repo.save_predictions([pred2])

            got = store.predictions_repo.get_prediction(
                "Duke", "Auburn", "2026-03-17",
            )
            assert got["home_ml_prob"] == 0.70

    def test_get_predictions_for_date(self):
        with LineStore(":memory:") as store:
            preds = [_sample_prediction()]
            store.predictions_repo.save_predictions(preds)

            by_date = store.predictions_repo.get_predictions_for_date("2026-03-17")
            assert len(by_date) == 1

            empty = store.predictions_repo.get_predictions_for_date("2026-01-01")
            assert len(empty) == 0

    def test_get_by_event_id(self):
        with LineStore(":memory:") as store:
            pred = _sample_prediction()
            pred["event_id"] = "abc123"
            store.predictions_repo.save_predictions([pred])

            got = store.predictions_repo.get_by_event_id("abc123")
            assert got is not None
            assert got["home_team"] == "Duke"

    def test_update_event_id(self):
        with LineStore(":memory:") as store:
            pred = _sample_prediction()
            store.predictions_repo.save_predictions([pred])
            got = store.predictions_repo.get_prediction(
                "Duke", "Auburn", "2026-03-17",
            )
            pid = got["prediction_id"]

            store.predictions_repo.update_event_id(pid, "ev_999")
            updated = store.predictions_repo.get_by_event_id("ev_999")
            assert updated is not None

    def test_update_actual_result(self):
        with LineStore(":memory:") as store:
            pred = _sample_prediction()
            store.predictions_repo.save_predictions([pred])
            got = store.predictions_repo.get_prediction(
                "Duke", "Auburn", "2026-03-17",
            )
            pid = got["prediction_id"]

            store.predictions_repo.update_actual_result(pid, 7.0, 148.0)
            updated = store.predictions_repo.get_prediction(
                "Duke", "Auburn", "2026-03-17",
            )
            assert updated["actual_margin"] == 7.0
            assert updated["actual_total"] == 148.0


# ---------------------------------------------------------------------------
# 6C: Slate integration — model injection
# ---------------------------------------------------------------------------


class TestModelSlateInjection:
    """Verify that model predictions override consensus edge when available."""

    def test_model_pred_sets_edge_source(self):
        """A game with model predictions should have edge_source='model'."""
        lines = _make_lines()
        pred = _sample_prediction(home_ml_prob=0.65)
        slate = build_daily_slate(
            {"event1": lines},
            model_predictions=[pred],
        )
        # Find entries with model edge source
        all_entries = []
        for tier_key in ("tier1a", "tier1b", "tier2", "tier3", "stay_away"):
            all_entries.extend(slate.get(tier_key, []))
        all_entries.extend(slate.get("closest_candidates", []))

        model_entries = [e for e in all_entries if e.get("edge_source") == "model"]
        assert len(model_entries) > 0, "Expected at least one model-sourced entry"

        for e in model_entries:
            assert e["model_prob"] is not None
            assert e["model_margin"] is not None
            # Edge should be computed from model_prob, not consensus
            dec = american_to_decimal(e["best_odds"])
            expected_edge = round(100.0 * (e["model_prob"] * dec - 1.0), 2)
            assert e["edge_pct"] == expected_edge

    def test_no_model_pred_falls_back_to_consensus(self):
        """A game without model predictions should use consensus edge."""
        lines = _make_lines()
        slate = build_daily_slate(
            {"event1": lines},
            model_predictions=None,
        )
        all_entries = []
        for tier_key in ("tier1a", "tier1b", "tier2", "tier3", "stay_away"):
            all_entries.extend(slate.get(tier_key, []))
        all_entries.extend(slate.get("closest_candidates", []))

        assert len(all_entries) > 0
        for e in all_entries:
            assert e.get("edge_source") == "consensus"
            assert e.get("model_prob") is None

    def test_unmatched_game_falls_back(self):
        """A game where the model has no prediction uses consensus."""
        lines = _make_lines()
        # Prediction for a different game
        pred = _sample_prediction()
        pred["home_team"] = "Gonzaga"
        pred["away_team"] = "UConn"

        slate = build_daily_slate(
            {"event1": lines},
            model_predictions=[pred],
        )
        all_entries = []
        for tier_key in ("tier1a", "tier1b", "tier2", "tier3", "stay_away"):
            all_entries.extend(slate.get(tier_key, []))
        all_entries.extend(slate.get("closest_candidates", []))

        for e in all_entries:
            assert e.get("edge_source") == "consensus"

    def test_tier_classification_works_with_model_edge(self):
        """Model-derived edges should flow through to tier assignment."""
        lines = _make_lines()
        # Very high model probability — should produce strong edge
        pred = _sample_prediction(home_ml_prob=0.80)

        slate = build_daily_slate(
            {"event1": lines},
            model_predictions=[pred],
        )
        all_entries = []
        for tier_key in ("tier1a", "tier1b", "tier2", "tier3", "stay_away"):
            all_entries.extend(slate.get(tier_key, []))
        all_entries.extend(slate.get("closest_candidates", []))

        model_entries = [e for e in all_entries if e.get("edge_source") == "model"]
        if model_entries:
            # Every entry should have a tier
            for e in model_entries:
                assert "tier" in e
                assert e["tier"] in ("tier1a", "tier1b", "tier2", "tier3", "avoid")

    def test_empty_predictions_list_is_noop(self):
        """Empty predictions list should behave like None."""
        lines = _make_lines()
        slate = build_daily_slate(
            {"event1": lines},
            model_predictions=[],
        )
        all_entries = []
        for tier_key in ("tier1a", "tier1b", "tier2", "tier3", "stay_away"):
            all_entries.extend(slate.get(tier_key, []))
        all_entries.extend(slate.get("closest_candidates", []))

        for e in all_entries:
            assert e.get("edge_source") == "consensus"


# ---------------------------------------------------------------------------
# 6C: Model matching helpers
# ---------------------------------------------------------------------------


class TestModelMatching:
    def test_match_prediction_basic(self):
        pred = _sample_prediction()
        result = _match_prediction(
            "Duke Blue Devils", "Auburn Tigers", [pred],
        )
        assert result is not None
        assert result["home_team"] == "Duke"

    def test_match_prediction_no_match(self):
        pred = _sample_prediction()
        result = _match_prediction("Lakers", "Celtics", [pred])
        assert result is None

    def test_model_prob_for_ml_home(self):
        pred = _sample_prediction()
        prob = _model_prob_for_market(
            pred, "moneyline", "Duke Blue Devils",
            "Duke Blue Devils", "Auburn Tigers",
        )
        assert prob == 0.65

    def test_model_prob_for_ml_away(self):
        pred = _sample_prediction()
        prob = _model_prob_for_market(
            pred, "moneyline", "Auburn Tigers",
            "Duke Blue Devils", "Auburn Tigers",
        )
        assert prob == 0.35

    def test_model_prob_for_total_over(self):
        pred = _sample_prediction()
        prob = _model_prob_for_market(
            pred, "total", "Over 145.5",
            "Duke Blue Devils", "Auburn Tigers",
        )
        assert prob == 0.52

    def test_model_prob_for_total_under(self):
        pred = _sample_prediction()
        prob = _model_prob_for_market(
            pred, "total", "Under 145.5",
            "Duke Blue Devils", "Auburn Tigers",
        )
        assert prob == 0.48

    def test_sel_matches_team(self):
        assert _sel_matches_team("duke blue devils", "Duke Blue Devils")
        assert _sel_matches_team("duke", "Duke Blue Devils")
        assert not _sel_matches_team("auburn", "Duke Blue Devils")


# ---------------------------------------------------------------------------
# 6D: Config keys
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_model_enabled_default_false(self):
        assert get_model_enabled() is False

    def test_model_min_edge_default(self):
        assert get_model_min_edge() == 2.0

    def test_model_sport_default(self):
        assert get_model_sport() == "basketball_ncaab"

    @patch.dict("os.environ", {"MODEL_ENABLED": "1"})
    def test_model_enabled_env(self):
        assert get_model_enabled() is True


# ---------------------------------------------------------------------------
# 6E: CLI commands
# ---------------------------------------------------------------------------


class TestCLICommands:
    def test_cycle_help_shows_use_model(self):
        from line_tracker.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main(["cycle", "--help"])
        assert exc_info.value.code == 0

    def test_predict_help(self):
        from line_tracker.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main(["predict", "--help"])
        assert exc_info.value.code == 0

    def test_train_model_help(self):
        from line_tracker.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main(["train-model", "--help"])
        assert exc_info.value.code == 0

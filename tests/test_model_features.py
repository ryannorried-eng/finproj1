"""Tests for line_tracker.model.features — feature engineering."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from line_tracker.model import features as mod

# ---------------------------------------------------------------------------
# Fixtures — synthetic team rating dicts
# ---------------------------------------------------------------------------

DUKE = {
    "team": "Duke",
    "adj_em": 37.35,
    "adj_oe": 128.16,
    "adj_de": 90.81,
    "adj_t": 65.80,
    "efg_o": 56.8,
    "efg_d": 46.2,
    "to_o": 15.7,
    "to_d": 18.1,
    "or_pct": 38.1,
    "ftr_o": 37.8,
    "ftr_d": 23.7,
    "sos": 13.68,
}

UNC = {
    "team": "North Carolina",
    "adj_em": 22.20,
    "adj_oe": 120.50,
    "adj_de": 98.30,
    "adj_t": 69.50,
    "efg_o": 53.0,
    "efg_d": 49.0,
    "to_o": 17.0,
    "to_d": 16.0,
    "or_pct": 32.0,
    "ftr_o": 33.0,
    "ftr_d": 29.0,
    "sos": 8.50,
}


# ---------------------------------------------------------------------------
# Tests — build_game_features
# ---------------------------------------------------------------------------


class TestBuildGameFeatures:
    def test_efficiency_gap(self):
        f = mod.build_game_features(DUKE, UNC)
        assert f["adj_em_diff"] == pytest.approx(37.35 - 22.20)
        assert f["adj_oe_diff"] == pytest.approx(128.16 - 120.50)
        assert f["adj_de_diff"] == pytest.approx(90.81 - 98.30)

    def test_tempo_features(self):
        f = mod.build_game_features(DUKE, UNC)
        assert f["tempo_avg"] == pytest.approx((65.80 + 69.50) / 2.0)
        assert f["tempo_diff"] == pytest.approx(65.80 - 69.50)
        assert f["tempo_mismatch"] == pytest.approx(abs(65.80 - 69.50))

    def test_four_factors_matchup(self):
        f = mod.build_game_features(DUKE, UNC)
        # efg_edge = home efg_o - away efg_d
        assert f["efg_edge"] == pytest.approx(56.8 - 49.0)
        # to_edge = away to_o - home to_o (positive = home forces more TOs)
        assert f["to_edge"] == pytest.approx(17.0 - 15.7)
        # or_edge = home or_pct - away or_pct
        assert f["or_edge"] == pytest.approx(38.1 - 32.0)
        # ftr_edge = home ftr_o - away ftr_d
        assert f["ftr_edge"] == pytest.approx(37.8 - 29.0)

    def test_home_court_default(self):
        f = mod.build_game_features(DUKE, UNC)
        assert f["home_court"] == 3.5
        assert f["tournament_flag"] == 0.0

    def test_neutral_site(self):
        f = mod.build_game_features(DUKE, UNC, neutral_site=True)
        assert f["home_court"] == 0.0
        assert f["tournament_flag"] == 0.0

    def test_tournament_neutral(self):
        f = mod.build_game_features(
            DUKE, UNC, neutral_site=True, tournament=True,
        )
        assert f["home_court"] == 1.5
        assert f["tournament_flag"] == 1.0

    def test_tournament_non_neutral(self):
        """Tournament flag set but not neutral — home court stays at 3.5."""
        f = mod.build_game_features(
            DUKE, UNC, neutral_site=False, tournament=True,
        )
        assert f["home_court"] == 3.5
        assert f["tournament_flag"] == 1.0

    def test_sos_diff(self):
        f = mod.build_game_features(DUKE, UNC)
        assert f["sos_diff"] == pytest.approx(13.68 - 8.50)

    def test_all_feature_columns_present(self):
        """Output dict should contain exactly the canonical feature columns."""
        f = mod.build_game_features(DUKE, UNC)
        assert set(f.keys()) == set(mod.FEATURE_COLUMNS)

    def test_symmetry(self):
        """Swapping home/away should flip signs on directional features."""
        f1 = mod.build_game_features(DUKE, UNC, neutral_site=True)
        f2 = mod.build_game_features(UNC, DUKE, neutral_site=True)
        assert f1["adj_em_diff"] == pytest.approx(-f2["adj_em_diff"])
        assert f1["adj_oe_diff"] == pytest.approx(-f2["adj_oe_diff"])
        assert f1["adj_de_diff"] == pytest.approx(-f2["adj_de_diff"])
        assert f1["tempo_diff"] == pytest.approx(-f2["tempo_diff"])
        assert f1["tempo_mismatch"] == pytest.approx(f2["tempo_mismatch"])
        assert f1["sos_diff"] == pytest.approx(-f2["sos_diff"])

    def test_specific_efg_edge_value(self):
        """If home AdjOE=110 (efg_o=110), away AdjDE=100 (efg_d=100),
        efg_edge should be 10."""
        home = {**DUKE, "efg_o": 110.0}
        away = {**UNC, "efg_d": 100.0}
        f = mod.build_game_features(home, away)
        assert f["efg_edge"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests — FEATURE_COLUMNS
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    def test_length(self):
        assert len(mod.FEATURE_COLUMNS) == 13

    def test_no_duplicates(self):
        assert len(mod.FEATURE_COLUMNS) == len(set(mod.FEATURE_COLUMNS))

    def test_matches_build_output(self):
        f = mod.build_game_features(DUKE, UNC)
        assert list(f.keys()) == mod.FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Tests — build_training_dataset
# ---------------------------------------------------------------------------

# Minimal fake DataFrames to exercise the builder without network calls.

_FAKE_RATINGS = pd.DataFrame([
    {
        "team": "Duke", "adj_em": 37.35, "adj_oe": 128.16,
        "adj_de": 90.81, "adj_t": 65.80, "efg_o": 56.8, "efg_d": 46.2,
        "to_o": 15.7, "to_d": 18.1, "or_pct": 38.1, "ftr_o": 37.8,
        "ftr_d": 23.7, "sos": 13.68,
    },
    {
        "team": "North Carolina", "adj_em": 22.20, "adj_oe": 120.50,
        "adj_de": 98.30, "adj_t": 69.50, "efg_o": 53.0, "efg_d": 49.0,
        "to_o": 17.0, "to_d": 16.0, "or_pct": 32.0, "ftr_o": 33.0,
        "ftr_d": 29.0, "sos": 8.50,
    },
])

_FAKE_GAMES = pd.DataFrame([
    {
        "date": pd.Timestamp("2026-01-15"),
        "home_team": "Duke",
        "away_team": "North Carolina",
        "home_score": 85,
        "away_score": 70,
        "margin": 15,
        "neutral_site": False,
        "game_id": "DukeNorth Carolina1-15",
    },
    {
        "date": pd.Timestamp("2026-02-10"),
        "home_team": "North Carolina",
        "away_team": "Duke",
        "home_score": 75,
        "away_score": 80,
        "margin": -5,
        "neutral_site": False,
        "game_id": "North CarolinaDuke2-10",
    },
])


class TestBuildTrainingDataset:
    def test_builds_features_and_target(self):
        with (
            patch.object(mod, "fetch_team_ratings", return_value=_FAKE_RATINGS),
            patch.object(mod, "fetch_game_results", return_value=_FAKE_GAMES),
        ):
            X, y = mod.build_training_dataset([2026])  # noqa: N806

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 2
        assert len(y) == 2
        assert list(X.columns) == mod.FEATURE_COLUMNS

        # First game: Duke home, margin=15
        assert y.iloc[0] == 15.0
        # Second game: UNC home, margin=-5
        assert y.iloc[1] == -5.0

    def test_feature_values_match_builder(self):
        """Training dataset features should equal direct build_game_features."""
        with (
            patch.object(mod, "fetch_team_ratings", return_value=_FAKE_RATINGS),
            patch.object(mod, "fetch_game_results", return_value=_FAKE_GAMES),
        ):
            X, _ = mod.build_training_dataset([2026])  # noqa: N806

        expected = mod.build_game_features(DUKE, UNC)
        for col in mod.FEATURE_COLUMNS:
            assert X.iloc[0][col] == pytest.approx(expected[col]), (
                f"Mismatch on {col}"
            )

    def test_skips_unknown_teams(self):
        """Games with teams not in ratings should be silently dropped."""
        games_with_unknown = pd.concat([
            _FAKE_GAMES,
            pd.DataFrame([{
                "date": pd.Timestamp("2026-03-01"),
                "home_team": "Duke",
                "away_team": "Unknown College",
                "home_score": 90,
                "away_score": 50,
                "margin": 40,
                "neutral_site": False,
                "game_id": "DukeUnknown3-01",
            }]),
        ], ignore_index=True)

        with (
            patch.object(mod, "fetch_team_ratings", return_value=_FAKE_RATINGS),
            patch.object(mod, "fetch_game_results", return_value=games_with_unknown),
        ):
            X, y = mod.build_training_dataset([2026])  # noqa: N806

        # Only the 2 known-team games should remain
        assert len(X) == 2
        assert len(y) == 2

    def test_multi_season(self):
        """Multiple seasons should concatenate correctly."""
        with (
            patch.object(mod, "fetch_team_ratings", return_value=_FAKE_RATINGS),
            patch.object(mod, "fetch_game_results", return_value=_FAKE_GAMES),
        ):
            X, y = mod.build_training_dataset([2025, 2026])  # noqa: N806

        # 2 games per season × 2 seasons
        assert len(X) == 4
        assert len(y) == 4

    def test_neutral_site_propagated(self):
        """Neutral-site games should have home_court=0.0."""
        neutral_game = pd.DataFrame([{
            "date": pd.Timestamp("2026-03-15"),
            "home_team": "Duke",
            "away_team": "North Carolina",
            "home_score": 78,
            "away_score": 72,
            "margin": 6,
            "neutral_site": True,
            "game_id": "DukeNorth Carolina3-15",
        }])

        with (
            patch.object(mod, "fetch_team_ratings", return_value=_FAKE_RATINGS),
            patch.object(mod, "fetch_game_results", return_value=neutral_game),
        ):
            X, _ = mod.build_training_dataset([2026])  # noqa: N806

        assert X.iloc[0]["home_court"] == 0.0

    def test_empty_season_returns_empty(self):
        """Season with no games should return empty DataFrame/Series."""
        empty_games = pd.DataFrame(
            columns=[
                "date", "home_team", "away_team", "home_score",
                "away_score", "margin", "neutral_site", "game_id",
            ],
        )

        with (
            patch.object(mod, "fetch_team_ratings", return_value=_FAKE_RATINGS),
            patch.object(mod, "fetch_game_results", return_value=empty_games),
        ):
            X, y = mod.build_training_dataset([2026])  # noqa: N806

        assert len(X) == 0
        assert len(y) == 0
        assert list(X.columns) == mod.FEATURE_COLUMNS

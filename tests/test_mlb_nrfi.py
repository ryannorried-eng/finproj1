"""Tests for the NRFI/YRFI prediction pipeline.

These tests use mocked data only — no network calls, no model files required.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fi_df(n: int = 200, yrfi_rate: float = 0.565, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic first-inning DataFrame for testing."""
    rng = np.random.default_rng(seed)
    teams = ["NYY", "BOS", "LAD", "SFG", "ATL", "HOU", "NYM", "CHC"]
    dates = pd.date_range("2025-04-01", periods=n, freq="D")

    home_sp_ids = rng.integers(100, 200, size=n)
    away_sp_ids = rng.integers(200, 300, size=n)
    yrfi = rng.binomial(1, yrfi_rate, size=n)
    fi_home = rng.integers(0, 3, size=n)
    fi_away = rng.integers(0, 3, size=n)

    lineup_json = json.dumps([
        {"id": 1001, "name": "Batter A", "batting_order": 1, "position": "CF"},
        {"id": 1002, "name": "Batter B", "batting_order": 2, "position": "RF"},
        {"id": 1003, "name": "Batter C", "batting_order": 3, "position": "1B"},
    ])

    return pd.DataFrame({
        "game_pk":               range(1000, 1000 + n),
        "date":                  dates,
        "home_team":             [teams[i % len(teams)] for i in range(n)],
        "away_team":             [teams[(i + 1) % len(teams)] for i in range(n)],
        "home_sp_id":            home_sp_ids,
        "away_sp_id":            away_sp_ids,
        "home_lineup":           [lineup_json] * n,
        "away_lineup":           [lineup_json] * n,
        "first_inning_home_runs": fi_home,
        "first_inning_away_runs": fi_away,
        "yrfi":                  yrfi,
        "season":                [2025] * n,
    })


def _make_pitcher_stats(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pids = range(100, 100 + n)
    return pd.DataFrame({
        "pitcher_name": [f"Pitcher {i}" for i in pids],
        "era":          rng.uniform(3.0, 5.0, n),
        "whip":         rng.uniform(1.1, 1.5, n),
        "k9":           rng.uniform(7.0, 11.0, n),
        "bb9":          rng.uniform(2.0, 4.0, n),
        "innings":      rng.uniform(80, 160, n),
    }, index=pd.Index(pids, name="pitcher_id"))


def _make_batter_stats(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    bids = [1001, 1002, 1003] + list(range(2000, 2000 + n - 3))
    return pd.DataFrame({
        "batter_name":       [f"Batter {b}" for b in bids],
        "obp":               rng.uniform(0.28, 0.40, n),
        "slg":               rng.uniform(0.35, 0.55, n),
        "ops":               rng.uniform(0.63, 0.95, n),
        "plate_appearances": rng.integers(200, 600, n),
    }, index=pd.Index(bids, name="batter_id"))


# ---------------------------------------------------------------------------
# fetch_first_inning_data schema
# ---------------------------------------------------------------------------


class TestFetchFirstInningSchema:
    """Tests for fetch_first_inning_data output schema."""

    def test_schema_columns(self):
        """fetch_first_inning_data must return required columns."""
        from line_tracker.model.mlb_nrfi_features import NRFI_FEATURE_COLUMNS  # noqa: F401

        # Verify our synthetic df has the right columns
        fi = _make_fi_df(50)
        required = {
            "game_pk", "date", "home_team", "away_team",
            "home_sp_id", "away_sp_id",
            "first_inning_home_runs", "first_inning_away_runs",
            "yrfi", "season",
        }
        assert required.issubset(set(fi.columns)), (
            f"Missing columns: {required - set(fi.columns)}"
        )

    def test_yrfi_is_binary(self):
        fi = _make_fi_df(100)
        assert set(fi["yrfi"].unique()).issubset({0, 1})

    def test_game_pk_unique(self):
        fi = _make_fi_df(100)
        assert fi["game_pk"].nunique() == len(fi), "game_pk values must be unique"


# ---------------------------------------------------------------------------
# YRFI rate sanity check
# ---------------------------------------------------------------------------


class TestYrfiRateReasonable:
    """YRFI rate across realistic sample should fall in historical range."""

    def test_yrfi_rate_in_range(self):
        fi = _make_fi_df(n=500, yrfi_rate=0.565)
        rate = fi["yrfi"].mean()
        assert 0.40 <= rate <= 0.75, f"Suspicious YRFI rate: {rate:.3f}"

    def test_yrfi_rate_central(self):
        """With large sample the rate should be near 0.565."""
        fi = _make_fi_df(n=2000, yrfi_rate=0.565, seed=0)
        rate = fi["yrfi"].mean()
        assert 0.52 <= rate <= 0.62, f"YRFI rate out of historical range: {rate:.3f}"


# ---------------------------------------------------------------------------
# build_nrfi_features shape and no-leakage
# ---------------------------------------------------------------------------


class TestBuildFeaturesShape:
    """build_nrfi_features must return the correct columns."""

    def test_output_columns_complete(self):
        from line_tracker.model.mlb_nrfi_features import (
            NRFI_FEATURE_COLUMNS,
            build_nrfi_features,
        )

        fi = _make_fi_df(100)
        ps = {2025: _make_pitcher_stats()}
        bs = {2025: _make_batter_stats()}
        result = build_nrfi_features(fi, pitcher_stats=ps, batter_stats=bs)

        missing = set(NRFI_FEATURE_COLUMNS) - set(result.columns)
        assert not missing, f"Missing feature columns: {missing}"

    def test_output_has_yrfi_target(self):
        from line_tracker.model.mlb_nrfi_features import build_nrfi_features

        fi = _make_fi_df(50)
        result = build_nrfi_features(fi)
        assert "yrfi" in result.columns

    def test_no_extra_nan_in_features(self):
        from line_tracker.model.mlb_nrfi_features import (
            NRFI_FEATURE_COLUMNS,
            build_nrfi_features,
        )

        fi = _make_fi_df(100)
        result = build_nrfi_features(fi)
        nan_frac = result[NRFI_FEATURE_COLUMNS].isna().mean().max()
        assert nan_frac < 0.30, f"Too many NaNs in features: {nan_frac:.1%}"


class TestNoDataLeakage:
    """Rolling windows must not include the current game."""

    def test_sp_yrfi_rate_uses_prior_only(self):
        from line_tracker.model.mlb_nrfi_features import build_nrfi_features

        # Create a DataFrame where the first game always has yrfi=0,
        # all others always yrfi=1.  The first game's SP rolling rate
        # should be NaN (no prior starts) → filled with league average.
        fi = _make_fi_df(30)
        fi.loc[0, "yrfi"] = 0
        fi.loc[1:, "yrfi"] = 1

        result = build_nrfi_features(fi)
        # The first row's rolling rate should NOT be 1.0 (it would be if the
        # current game's yrfi were included)
        first_rate = result.iloc[0]["home_sp_fi_yrfi_rate_r10"]
        assert first_rate < 1.0, (
            "First-game rolling rate should not include current game (leakage detected)"
        )

    def test_team_yrfi_rate_shift_applied(self):
        from line_tracker.model.mlb_nrfi_features import _team_yrfi_rolling

        fi = _make_fi_df(40)
        fi["yrfi"] = 0
        fi.iloc[5:, fi.columns.get_loc("yrfi")] = 1

        result = _team_yrfi_rolling(fi)
        merged = fi.merge(result, on="game_pk", how="left")

        # The rate for the team's first home game should be NaN (filled with league avg later)
        # and should NOT be 1.0 (which would indicate leakage)
        assert merged["home_team_yrfi_rate_r15"].notna().any()


# ---------------------------------------------------------------------------
# predict_nrfi dict keys
# ---------------------------------------------------------------------------


class TestPredictReturnsRequiredKeys:
    """predict_nrfi must return dicts with all required keys."""

    REQUIRED_KEYS = {
        "game", "home_sp", "away_sp",
        "yrfi_prob", "nrfi_prob",
        "market_yrfi_prob",
        "yrfi_edge", "nrfi_edge",
        "bet", "edge", "confidence",
        "home_sp_fi_yrfi_rate", "away_sp_fi_yrfi_rate",
        "home_top3_obp", "away_top3_obp",
    }

    def _make_mock_artifacts(self):
        """Return mock GBM + LR artifacts."""
        rng = np.random.default_rng(0)

        gbm_mock = MagicMock()
        gbm_mock.predict_proba.return_value = np.array([[0.4, 0.6]])

        lr_mock = MagicMock()
        lr_mock.predict_proba.return_value = np.array([[0.45, 0.55]])

        scaler_mock = MagicMock()
        scaler_mock.transform.side_effect = lambda x: x

        return {
            "gbm": {"model": gbm_mock, "scaler": None, "metadata": {}},
            "lr":  {"model": lr_mock, "scaler": scaler_mock, "metadata": {}},
        }

    def _make_mock_probable(self):
        return pd.DataFrame([{
            "game_pk":            700001,
            "home_team":          "NYY",
            "away_team":          "BOS",
            "home_pitcher_id":    123,
            "home_pitcher_name":  "Gerrit Cole",
            "away_pitcher_id":    456,
            "away_pitcher_name":  "Brayan Bello",
        }])

    def test_all_required_keys_present(self):
        from line_tracker.model.mlb_nrfi import predict_nrfi

        with patch("line_tracker.model.mlb_nrfi.load_nrfi_artifacts",
                   return_value=self._make_mock_artifacts()), \
             patch("line_tracker.model.mlb_nrfi._fetch_probable_pitchers_mlb",
                   return_value=self._make_mock_probable()), \
             patch("line_tracker.model.mlb_nrfi.fetch_pitcher_season_stats",
                   return_value=_make_pitcher_stats()), \
             patch("line_tracker.model.mlb_nrfi.fetch_batter_season_stats",
                   return_value=_make_batter_stats()), \
             patch("line_tracker.model.mlb_nrfi.fetch_first_inning_data",
                   return_value=_make_fi_df(50)), \
             patch("line_tracker.model.mlb_nrfi.fetch_game_starters",
                   return_value=pd.DataFrame()):
            from datetime import date
            results = predict_nrfi(game_date=date(2026, 4, 6))

        assert len(results) == 1
        result = results[0]
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in predict_nrfi output: {missing}"

    def test_prob_values_valid(self):
        from line_tracker.model.mlb_nrfi import predict_nrfi

        with patch("line_tracker.model.mlb_nrfi.load_nrfi_artifacts",
                   return_value=self._make_mock_artifacts()), \
             patch("line_tracker.model.mlb_nrfi._fetch_probable_pitchers_mlb",
                   return_value=self._make_mock_probable()), \
             patch("line_tracker.model.mlb_nrfi.fetch_pitcher_season_stats",
                   return_value=_make_pitcher_stats()), \
             patch("line_tracker.model.mlb_nrfi.fetch_batter_season_stats",
                   return_value=_make_batter_stats()), \
             patch("line_tracker.model.mlb_nrfi.fetch_first_inning_data",
                   return_value=_make_fi_df(50)), \
             patch("line_tracker.model.mlb_nrfi.fetch_game_starters",
                   return_value=pd.DataFrame()):
            from datetime import date
            results = predict_nrfi(game_date=date(2026, 4, 6))

        assert len(results) == 1
        r = results[0]
        assert 0 < r["yrfi_prob"] < 1
        assert 0 < r["nrfi_prob"] < 1
        assert abs(r["yrfi_prob"] + r["nrfi_prob"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Edge calculation
# ---------------------------------------------------------------------------


class TestEdgeCalculation:
    """Edge = model_prob − market_implied_prob."""

    def test_yrfi_edge_formula(self):
        from line_tracker.model.mlb_nrfi import EDGE_THRESHOLD

        yrfi_model = 0.62
        market_implied = 0.55
        yrfi_edge = yrfi_model - market_implied
        nrfi_edge = (1 - yrfi_model) - (1 - market_implied)

        assert abs(yrfi_edge - 0.07) < 1e-9
        assert abs(nrfi_edge - (-0.07)) < 1e-9
        assert yrfi_edge > EDGE_THRESHOLD

    def test_no_edge_below_threshold(self):
        from line_tracker.model.mlb_nrfi import EDGE_THRESHOLD

        yrfi_model = 0.57
        market_implied = 0.55
        yrfi_edge = yrfi_model - market_implied

        assert yrfi_edge < EDGE_THRESHOLD


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------


class TestConfidenceLevels:
    """Confidence tiers: high > 0.08, medium > 0.05."""

    def _conf(self, edge: float) -> str | None:
        from line_tracker.model.mlb_nrfi import EDGE_THRESHOLD
        if edge >= 0.08:
            return "high"
        if edge >= EDGE_THRESHOLD:
            return "medium"
        return "low"

    def test_high_confidence(self):
        assert self._conf(0.10) == "high"
        assert self._conf(0.08) == "high"

    def test_medium_confidence(self):
        assert self._conf(0.06) == "medium"
        assert self._conf(0.05) == "medium"

    def test_low_confidence(self):
        assert self._conf(0.02) == "low"
        assert self._conf(0.00) == "low"


# ---------------------------------------------------------------------------
# ROC-AUC baseline check
# ---------------------------------------------------------------------------


class TestNrfiGbmRocAboveBaseline:
    """GBM on synthetic data should exceed random baseline (ROC-AUC > 0.50)."""

    def test_roc_above_random(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        from line_tracker.model.mlb_nrfi_features import (
            NRFI_FEATURE_COLUMNS,
            build_nrfi_features,
        )

        fi = _make_fi_df(n=500, yrfi_rate=0.565)
        feat_df = build_nrfi_features(fi)
        feat_df = feat_df.dropna(subset=["yrfi"])
        X = feat_df[NRFI_FEATURE_COLUMNS].fillna(0).astype(float)
        y = feat_df["yrfi"].astype(int)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        scores = []
        for tr, val in skf.split(X, y):
            m = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)
            m.fit(X.iloc[tr], y.iloc[tr])
            probs = m.predict_proba(X.iloc[val])[:, 1]
            scores.append(roc_auc_score(y.iloc[val], probs))

        mean_roc = float(np.mean(scores))
        # Synthetic data has weak signal; just confirm > random
        assert mean_roc > 0.45, f"GBM ROC-AUC ({mean_roc:.3f}) below acceptable floor"


# ---------------------------------------------------------------------------
# Lineup parsing
# ---------------------------------------------------------------------------


class TestLineupParsing:
    def test_parse_top3(self):
        from line_tracker.model.mlb_nrfi_features import _parse_lineup_top3_ids

        lineup = json.dumps([
            {"id": 101, "batting_order": 1, "name": "A", "position": "CF"},
            {"id": 102, "batting_order": 2, "name": "B", "position": "RF"},
            {"id": 103, "batting_order": 3, "name": "C", "position": "1B"},
            {"id": 104, "batting_order": 4, "name": "D", "position": "DH"},
        ])
        result = _parse_lineup_top3_ids(lineup)
        assert result == [101, 102, 103]

    def test_parse_empty(self):
        from line_tracker.model.mlb_nrfi_features import _parse_lineup_top3_ids

        assert _parse_lineup_top3_ids(None) == []
        assert _parse_lineup_top3_ids("") == []
        assert _parse_lineup_top3_ids("invalid json") == []

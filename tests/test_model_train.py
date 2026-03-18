"""Tests for the NCAAB model training module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from line_tracker.model import train as mod
from line_tracker.model.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)
_N_TRAIN = 500
_N_VAL = 150

# Ground truth: margin ≈ 1.0 * adj_em_diff + 3.5 * home_court + noise
_NOISE_STD = 8.0


def _make_features(n: int) -> pd.DataFrame:
    """Generate synthetic feature rows."""
    data = {}
    for col in FEATURE_COLUMNS:
        if col == "home_court":
            data[col] = _RNG.choice([0.0, 1.5, 3.5], size=n)
        elif col == "tournament_flag":
            data[col] = _RNG.choice([0.0, 1.0], size=n, p=[0.9, 0.1])
        else:
            data[col] = _RNG.normal(0, 5, size=n)
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def _make_target(X: pd.DataFrame) -> pd.Series:  # noqa: N803
    """Generate synthetic margin = adj_em_diff + 3.5*home_court + noise."""
    signal = X["adj_em_diff"].values + 3.5 * X["home_court"].values
    noise = _RNG.normal(0, _NOISE_STD, size=len(X))
    return pd.Series(signal + noise, name="margin")


@pytest.fixture()
def synthetic_data():
    X_train = _make_features(_N_TRAIN)  # noqa: N806
    y_train = _make_target(X_train)
    X_val = _make_features(_N_VAL)  # noqa: N806
    y_val = _make_target(X_val)
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Tests: evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_basic_metrics(self, synthetic_data):
        X_train, y_train, X_val, y_val = synthetic_data  # noqa: N806
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_train)  # noqa: N806
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        model.fit(X_s, y_train)

        result = mod.evaluate_model(model, scaler, X_val, y_val)
        assert "rmse" in result
        assert "mae" in result
        assert "mean_error" in result
        assert "calibration" in result
        # Model should learn the signal well enough for RMSE < 2*noise
        assert result["rmse"] < _NOISE_STD * 2

    def test_ats_accuracy_with_spreads(self, synthetic_data):
        X_train, y_train, X_val, y_val = synthetic_data  # noqa: N806
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        model = RidgeCV(alphas=[1.0])
        model.fit(scaler.fit_transform(X_train), y_train)

        # Use model predictions +/- small noise as "market spreads"
        preds = model.predict(scaler.transform(X_val))
        spreads = pd.Series(preds + _RNG.normal(0, 2, size=len(preds)))

        result = mod.evaluate_model(model, scaler, X_val, y_val, spreads)
        assert "ats_accuracy" in result
        assert 0.0 <= result["ats_accuracy"] <= 1.0

    def test_ats_without_spreads(self, synthetic_data):
        X_train, y_train, X_val, y_val = synthetic_data  # noqa: N806
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        model = RidgeCV(alphas=[1.0])
        model.fit(scaler.fit_transform(X_train), y_train)

        result = mod.evaluate_model(model, scaler, X_val, y_val)
        assert "ats_accuracy" not in result


# ---------------------------------------------------------------------------
# Tests: train_margin_model (mocked data pipeline)
# ---------------------------------------------------------------------------


def _mock_build_training_dataset(seasons):
    """Return synthetic data for any requested seasons."""
    n_per_season = 200
    total = n_per_season * len(seasons)
    X = _make_features(total)  # noqa: N806
    y = _make_target(X)
    return X, y


class TestTrainMarginModel:
    def test_ridge_default(self, tmp_path):
        """Ridge model trains, saves, and returns expected keys."""
        with (
            patch.object(
                mod, "build_training_dataset",
                side_effect=_mock_build_training_dataset,
            ),
            patch.object(mod, "_MODELS_DIR", tmp_path),
            patch.object(mod, "current_season", return_value=2026),
        ):
            result = mod.train_margin_model(
                seasons=[2023, 2024, 2025, 2026],
                model_type="ridge",
            )

        assert "model_path" in result
        assert "val_rmse" in result
        assert "val_mae" in result
        assert "n_train" in result
        assert "n_val" in result
        assert "feature_importances" in result
        assert "trained_at" in result
        assert result["n_train"] > 0
        assert result["n_val"] > 0
        assert Path(result["model_path"]).exists()
        # Ridge should fit the signal: RMSE < 2 * noise std
        assert result["val_rmse"] < _NOISE_STD * 2

    def test_model_learns_adj_em_diff(self, tmp_path):
        """The strongest feature should be adj_em_diff."""
        with (
            patch.object(
                mod, "build_training_dataset",
                side_effect=_mock_build_training_dataset,
            ),
            patch.object(mod, "_MODELS_DIR", tmp_path),
            patch.object(mod, "current_season", return_value=2026),
        ):
            result = mod.train_margin_model(
                seasons=[2023, 2024, 2025, 2026],
            )

        importances = result["feature_importances"]
        top_feature = max(importances, key=importances.get)
        # adj_em_diff or home_court should be top (both are in the signal)
        assert top_feature in ("adj_em_diff", "home_court")

    def test_default_seasons(self, tmp_path):
        """With no seasons arg, uses last 4 completed seasons."""
        captured_seasons = []

        def capture(seasons):
            captured_seasons.extend(seasons)
            return _mock_build_training_dataset(seasons)

        with (
            patch.object(mod, "build_training_dataset", side_effect=capture),
            patch.object(mod, "_MODELS_DIR", tmp_path),
            patch.object(mod, "current_season", return_value=2026),
        ):
            mod.train_margin_model()

        # First call is full dataset [2022,2023,2024,2025], second is val [2025]
        assert 2022 in captured_seasons
        assert 2025 in captured_seasons

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            mod.train_margin_model(model_type="random_forest")


# ---------------------------------------------------------------------------
# Tests: load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_load_specific_path(self, tmp_path):
        """load_model with explicit path returns saved artifacts."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        model = Ridge()
        X = _make_features(50)  # noqa: N806
        y = _make_target(X)
        scaler = StandardScaler()
        model.fit(scaler.fit_transform(X), y)

        path = tmp_path / "test_model.joblib"
        import joblib

        joblib.dump(
            {"model": model, "scaler": scaler, "metadata": {"test": True}},
            path,
        )

        m, s, meta = mod.load_model(str(path))
        assert hasattr(m, "predict")
        assert hasattr(s, "transform")
        assert meta["test"] is True

    def test_load_latest(self, tmp_path):
        """load_model with None loads latest model file."""
        import joblib
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        for i, ts in enumerate(["20260101_000000", "20260301_120000"]):
            path = tmp_path / f"ncaab_margin_ridge_v{ts}.joblib"
            joblib.dump(
                {
                    "model": Ridge(),
                    "scaler": StandardScaler(),
                    "metadata": {"version": i},
                },
                path,
            )

        with patch.object(mod, "_MODELS_DIR", tmp_path):
            _, _, meta = mod.load_model()

        assert meta["version"] == 1  # latest

    def test_load_no_models(self, tmp_path):
        with (
            patch.object(mod, "_MODELS_DIR", tmp_path),
            patch.object(mod, "_PROJECT_MODELS_DIR", tmp_path),
            pytest.raises(FileNotFoundError, match="No trained models"),
        ):
            mod.load_model()


# ---------------------------------------------------------------------------
# Tests: _feature_importances
# ---------------------------------------------------------------------------


class TestFeatureImportances:
    def test_linear_model(self):
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = _make_features(50)  # noqa: N806
        y = _make_target(X)
        from sklearn.preprocessing import StandardScaler

        s = StandardScaler()
        model.fit(s.fit_transform(X), y)

        imp = mod._feature_importances(model, list(FEATURE_COLUMNS))
        assert len(imp) == len(FEATURE_COLUMNS)
        assert all(v >= 0 for v in imp.values())
        assert sum(imp.values()) == pytest.approx(1.0, abs=0.01)

    def test_no_importances(self):
        """Model with no coef_ or feature_importances_ returns empty dict."""

        class DummyModel:
            pass

        imp = mod._feature_importances(DummyModel(), ["a", "b"])
        assert imp == {}


# ---------------------------------------------------------------------------
# Tests: _calibration_buckets
# ---------------------------------------------------------------------------


class TestCalibrationBuckets:
    def test_buckets_structure(self):
        preds = np.array([0.0, 5.0, -10.0, 20.0])
        actuals = np.array([1.0, 3.0, -8.0, 18.0])
        buckets = mod._calibration_buckets(preds, actuals)
        assert len(buckets) > 0
        for b in buckets:
            assert "bucket" in b
            assert "n" in b
            assert "pred_mean" in b
            assert "actual_mean" in b

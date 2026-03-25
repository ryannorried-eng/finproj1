"""
MLB model training — v1 team-form model.
Trains three models:
  mlb_moneyline — LogisticRegression → P(home wins)
  mlb_margin    — RidgeCV → predicted run differential
  mlb_totals    — RidgeCV → predicted total runs
Artifacts saved to models/ directory at repo root.
Naming: mlb_{model_type}_v{YYYYMMDD_HHMMSS}.joblib
Format: {"model": fitted, "scaler": StandardScaler, "metadata": {...}}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from line_tracker.model import mlb_data, mlb_features

logger = logging.getLogger(__name__)

_PROJECT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_HOME_MODELS_DIR = Path.home() / ".line_tracker" / "models"
_CLOUD_MODELS_DIR = Path("/mount/src/finproj1/models")


def train_mlb_models(
    seasons: list[int] | None = None,
    models_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict:
    """Train moneyline, margin, and totals models on MLB game data.

    Parameters
    ----------
    seasons:
        Seasons to include. Defaults to [2022, 2023, 2024, 2025].
    models_dir:
        Directory to save model artifacts. Defaults to repo-root/models/.
    force_refresh:
        Re-download pybaseball data even if cached.

    Returns
    -------
    dict: {"moneyline": path, "margin": path, "totals": path}
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    if models_dir is None:
        models_dir = _PROJECT_MODELS_DIR
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data and build features
    # ------------------------------------------------------------------
    df = mlb_data.load_mlb_training_data(seasons=seasons, force_refresh=force_refresh)
    if df.empty:
        msg = "No MLB training data loaded. Check pybaseball connectivity."
        raise RuntimeError(msg)

    X, y = mlb_features.build_feature_matrix(df)
    if X.empty:
        msg = "Feature matrix is empty. Check data quality."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Defensive numeric cleanup
    # ------------------------------------------------------------------
    X = X.astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    assert all(np.isscalar(v) for v in X.iloc[0]), "Non-scalar values in feature matrix"

    print(f"Number of features: {X.shape[1]}")
    print("Sample X.head():")
    print(X.head())

    n_games = len(X)
    feature_names = list(X.columns)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    tscv = TimeSeriesSplit(n_splits=5)
    result_paths: dict = {}

    # ------------------------------------------------------------------
    # Train moneyline model
    # ------------------------------------------------------------------
    print(f"Training mlb_moneyline on {n_games} games across {seasons}...")
    ml_model, ml_scaler, ml_cv_score = _train_moneyline(X, y["home_win"], tscv)
    ml_path = _save_artifact(
        model=ml_model,
        scaler=ml_scaler,
        model_type="moneyline",
        cv_score=ml_cv_score,
        cv_metric="accuracy",
        seasons=seasons,
        n_games=n_games,
        feature_names=feature_names,
        timestamp=timestamp,
        models_dir=models_dir,
    )
    result_paths["moneyline"] = ml_path

    # ------------------------------------------------------------------
    # Train margin model
    # ------------------------------------------------------------------
    print("Training mlb_margin...")
    mg_model, mg_scaler, mg_cv_score = _train_margin(X, y["run_diff"], tscv)
    mg_path = _save_artifact(
        model=mg_model,
        scaler=mg_scaler,
        model_type="margin",
        cv_score=mg_cv_score,
        cv_metric="mae",
        seasons=seasons,
        n_games=n_games,
        feature_names=feature_names,
        timestamp=timestamp,
        models_dir=models_dir,
    )
    result_paths["margin"] = mg_path

    # ------------------------------------------------------------------
    # Train totals model
    # ------------------------------------------------------------------
    print("Training mlb_totals...")
    tot_model, tot_scaler, tot_cv_score = _train_totals(X, y["total_runs"], tscv)
    tot_path = _save_artifact(
        model=tot_model,
        scaler=tot_scaler,
        model_type="totals",
        cv_score=tot_cv_score,
        cv_metric="mae",
        seasons=seasons,
        n_games=n_games,
        feature_names=feature_names,
        timestamp=timestamp,
        models_dir=models_dir,
    )
    result_paths["totals"] = tot_path

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _print_summary(result_paths, ml_cv_score, mg_cv_score, tot_cv_score, n_games)

    return result_paths


def _train_moneyline(X, y_target, tscv):
    """Train LogisticRegression with TimeSeriesSplit CV."""
    X_arr = X.values
    y_arr = y_target.values

    cv_scores = []
    for train_idx, val_idx in tscv.split(X_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)
        cv_scores.append(accuracy_score(y_val, preds))

    cv_score = float(np.mean(cv_scores))

    # Refit on full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_scaled, y_arr)

    return model, scaler, cv_score


def _train_regression(X, y_target, tscv, alphas):
    """Train RidgeCV with TimeSeriesSplit CV (kept for moneyline fallback)."""
    X_arr = X.values
    y_arr = y_target.values

    cv_maes = []
    for train_idx, val_idx in tscv.split(X_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = RidgeCV(alphas=alphas)
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)
        cv_maes.append(mean_absolute_error(y_val, preds))

    cv_score = float(np.mean(cv_maes))

    # Refit on full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    model = RidgeCV(alphas=alphas)
    model.fit(X_scaled, y_arr)

    return model, scaler, cv_score


def _train_totals(X, y_totals, tscv):
    """Train HistGradientBoostingRegressor for total runs with TimeSeriesSplit CV."""
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr  = X.iloc[train_idx].astype(float)
        X_val = X.iloc[val_idx].astype(float)
        y_tr  = y_totals.iloc[train_idx].astype(float)
        y_val = y_totals.iloc[val_idx].astype(float)

        y_tr = y_tr.clip(3, 15)
        y_val = y_val.clip(3, 15)

        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
        )
        model.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        scores.append(mae)

    # Refit on full dataset
    final_model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )
    y_totals_clipped = y_totals.astype(float).clip(3, 15)
    final_model.fit(X.astype(float), y_totals_clipped)

    # HistGradientBoosting does not use a scaler — return None
    return final_model, None, float(np.mean(scores))


def _train_margin(X, y_margin, tscv):
    """Train HistGradientBoostingRegressor for run margin with TimeSeriesSplit CV."""
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr  = X.iloc[train_idx].astype(float)
        X_val = X.iloc[val_idx].astype(float)
        y_tr  = y_margin.iloc[train_idx].astype(float)
        y_val = y_margin.iloc[val_idx].astype(float)

        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
        )
        model.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        scores.append(mae)

    final_model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )
    final_model.fit(X.astype(float), y_margin.astype(float))

    return final_model, None, float(np.mean(scores))


def _save_artifact(
    model,
    scaler: StandardScaler,
    model_type: str,
    cv_score: float,
    cv_metric: str,
    seasons: list[int],
    n_games: int,
    feature_names: list[str],
    timestamp: str,
    models_dir: Path,
) -> str:
    filename = f"mlb_{model_type}_v{timestamp}.joblib"
    path = models_dir / filename

    metadata = {
        "sport": "baseball_mlb",
        "model_type": model_type,
        "version": "v4_lineup_strength",
        "seasons_trained": seasons,
        "n_games": n_games,
        "feature_names": feature_names,
        "cv_score": cv_score,
        "cv_metric": cv_metric,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump({"model": model, "scaler": scaler, "metadata": metadata}, path)
    logger.info("Saved %s artifact to %s", model_type, path)
    return str(path)


def _print_summary(
    paths: dict,
    ml_acc: float,
    mg_mae: float,
    tot_mae: float,
    n_games: int,
) -> None:
    print("\n" + "=" * 70)
    print("MLB Model Training Summary")
    print("=" * 70)
    hdr = f"{'Model':<18} | {'CV Score':<12} | {'N Games':<8} | Artifact"
    print(hdr)
    print("-" * 70)
    print(
        f"{'mlb_moneyline':<18} | {ml_acc:.1%}{'':>6} | {n_games:<8} | "
        f"{Path(paths['moneyline']).name}"
    )
    print(
        f"{'mlb_margin':<18} | {'MAE ' + f'{mg_mae:.2f}':<12} | {n_games:<8} | "
        f"{Path(paths['margin']).name}"
    )
    print(
        f"{'mlb_totals':<18} | {'MAE ' + f'{tot_mae:.2f}':<12} | {n_games:<8} | "
        f"{Path(paths['totals']).name}"
    )
    print("=" * 70 + "\n")

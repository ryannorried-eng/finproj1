"""Train and evaluate NCAAB point-margin prediction models.

Supports Ridge regression (default) and XGBoost (optional).  Models are
persisted to ``~/.line_tracker/models/`` with a fitted scaler and metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from line_tracker.core.logging import get_logger
from line_tracker.model.data import current_season
from line_tracker.model.features import FEATURE_COLUMNS, build_training_dataset

log = get_logger(__name__)

_MODELS_DIR = Path.home() / ".line_tracker" / "models"
_PROJECT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    scaler: StandardScaler,
    X_val: pd.DataFrame,  # noqa: N803
    y_val: pd.Series,
    spreads_val: pd.Series | None = None,
) -> dict:
    """Compute validation metrics for a trained model.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible estimator.
    scaler:
        Fitted ``StandardScaler`` used during training.
    X_val:
        Raw (unscaled) validation features.
    y_val:
        Actual home-team point margins.
    spreads_val:
        Market closing spreads (home-team perspective) if available.

    Returns
    -------
    dict with rmse, mae, mean_error, and optionally ats_accuracy / calibration.
    """
    X_scaled = scaler.transform(X_val)  # noqa: N806
    preds = model.predict(X_scaled)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    mean_error = float(np.mean(preds - y_val.values))

    result: dict = {
        "rmse": rmse,
        "mae": mae,
        "mean_error": mean_error,
    }

    # ATS accuracy
    if spreads_val is not None:
        spreads = spreads_val.values
        actuals = y_val.values
        # Sign convention: spread is negative when home is favoured (standard
        # sportsbook notation).  Home covers when margin + spread > 0.
        model_pick = preds + spreads > 0
        actual_cover = actuals + spreads > 0
        # Exclude pushes (margin + spread == 0)
        not_push = actuals + spreads != 0
        if not_push.sum() > 0:
            correct = (model_pick == actual_cover) & not_push
            result["ats_accuracy"] = float(correct.sum() / not_push.sum())
            result["ats_n_games"] = int(not_push.sum())

    # Calibration by predicted margin buckets
    result["calibration"] = _calibration_buckets(preds, y_val.values)

    return result


def _calibration_buckets(
    preds: np.ndarray, actuals: np.ndarray,
) -> list[dict]:
    """Bin predictions and report actual vs predicted averages."""
    edges = [-np.inf, -15, -10, -5, -2, 2, 5, 10, 15, np.inf]
    labels = [
        "<-15", "-15:-10", "-10:-5", "-5:-2",
        "-2:2", "2:5", "5:10", "10:15", ">15",
    ]
    buckets = []
    for i in range(len(edges) - 1):
        mask = (preds >= edges[i]) & (preds < edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket": labels[i],
            "n": n,
            "pred_mean": float(preds[mask].mean()),
            "actual_mean": float(actuals[mask].mean()),
        })
    return buckets


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def _feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importances from a fitted model."""
    if hasattr(model, "coef_"):
        # Linear model — use absolute coefficient magnitude
        coefs = np.abs(model.coef_)
        total = coefs.sum() if coefs.sum() > 0 else 1.0
        return {
            name: float(coefs[i] / total)
            for i, name in enumerate(feature_names)
        }
    if hasattr(model, "feature_importances_"):
        # Tree-based model
        imp = model.feature_importances_
        return {
            name: float(imp[i])
            for i, name in enumerate(feature_names)
        }
    return {}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_margin_model(
    seasons: list[int] | None = None,
    *,
    model_type: str = "ridge",
) -> dict:
    """Train a point-margin prediction model on historical NCAAB games.

    Parameters
    ----------
    seasons:
        Seasons to include.  Defaults to the last 4 completed seasons.
    model_type:
        ``"ridge"`` (default) or ``"xgboost"``.

    Returns
    -------
    dict with model_path, val_rmse, val_mae, val_ats_accuracy, n_train,
    n_val, feature_importances, trained_at.
    """
    cur = current_season()
    if seasons is None:
        seasons = list(range(cur - 4, cur))

    if model_type not in ("ridge", "xgboost"):
        msg = f"Unsupported model_type: {model_type!r}"
        raise ValueError(msg)

    log.info(
        "Training %s model on seasons %s (val=%d)",
        model_type, seasons[:-1], seasons[-1],
    )

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    X, y = build_training_dataset(seasons)  # noqa: N806

    # Train/val split: most recent season = validation
    val_season = seasons[-1]
    # Compute how many games are in each season to find the split point.
    # build_training_dataset concatenates seasons in order, so the last
    # season's games are at the end.
    _, y_val_only = build_training_dataset([val_season])
    n_val = len(y_val_only)
    n_train = len(y) - n_val

    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]  # noqa: N806
    y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]

    log.info("Train: %d games, Val: %d games", n_train, n_val)

    # ------------------------------------------------------------------
    # Scale features
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # noqa: N806
    X_val_scaled = scaler.transform(X_val)  # noqa: N806

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    if model_type == "ridge":
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        model.fit(X_train_scaled, y_train)
        log.info("RidgeCV selected alpha=%.2f", model.alpha_)
    else:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            msg = (
                "xgboost is not installed. "
                "Install with: pip install 'line-tracker[xgboost]'"
            )
            raise ImportError(msg) from exc

        model = XGBRegressor(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    metrics = evaluate_model(model, scaler, X_val, y_val)

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------
    importances = _feature_importances(model, list(FEATURE_COLUMNS))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = (
        _MODELS_DIR / f"ncaab_margin_{model_type}_v{timestamp}.joblib"
    )

    metadata = {
        "model_type": model_type,
        "seasons": seasons,
        "val_season": val_season,
        "n_train": n_train,
        "n_val": n_val,
        "feature_columns": list(FEATURE_COLUMNS),
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_mean_error": metrics["mean_error"],
        "val_ats_accuracy": metrics.get("ats_accuracy"),
        "feature_importances": importances,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if model_type == "ridge" and hasattr(model, "alpha_"):
        metadata["ridge_alpha"] = float(model.alpha_)

    joblib.dump({"model": model, "scaler": scaler, "metadata": metadata}, model_path)
    log.info("Saved model to %s", model_path)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _print_summary(metadata, importances, model_path)

    return {
        "model_path": str(model_path),
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_ats_accuracy": metrics.get("ats_accuracy"),
        "n_train": n_train,
        "n_val": n_val,
        "feature_importances": importances,
        "trained_at": metadata["trained_at"],
    }


def _print_summary(
    metadata: dict, importances: dict[str, float], model_path: Path,
) -> None:
    print("\n" + "=" * 60)
    print("NCAAB Margin Model — Training Summary")
    print("=" * 60)
    print(
        f"Model trained on {metadata['n_train']} games, "
        f"validated on {metadata['n_val']} games."
    )
    print(f"  Val RMSE: {metadata['val_rmse']:.3f}")
    print(f"  Val MAE:  {metadata['val_mae']:.3f}")
    print(f"  Val bias: {metadata['val_mean_error']:+.3f}")
    ats = metadata.get("val_ats_accuracy")
    if ats is not None:
        print(f"  Val ATS accuracy: {ats:.1%}")
    print("\nTop 5 features by importance:")
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, imp in top5:
        print(f"  {name:20s} {imp:.3f}")
    print(f"\nModel saved to: {model_path}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_path: str | None = None,
) -> tuple:
    """Load a trained model, scaler, and metadata.

    Parameters
    ----------
    model_path:
        Path to ``.joblib`` file.  If *None*, loads the most recent model
        from ``~/.line_tracker/models/``.

    Returns
    -------
    (model, scaler, metadata) tuple.
    """
    if model_path is None:
        candidates = sorted(_MODELS_DIR.glob("ncaab_margin_*.joblib"))
        if not candidates:
            # Fallback: check models/ directory in the project root so that
            # deployed environments (e.g. Streamlit Cloud) can find a model
            # committed to the repo.
            candidates = sorted(
                _PROJECT_MODELS_DIR.glob("ncaab_margin_*.joblib"),
            )
        if not candidates:
            msg = f"No trained models found in {_MODELS_DIR}"
            raise FileNotFoundError(msg)
        model_path = str(candidates[-1])

    data = joblib.load(model_path)
    return data["model"], data["scaler"], data["metadata"]

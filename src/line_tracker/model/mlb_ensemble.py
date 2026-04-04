"""
MLB ensemble meta-model — stacks moneyline, margin, and totals base models.

Generates out-of-fold (OOF) predictions from all three base models, then trains
two meta-learners (LogisticRegression and GradientBoostingClassifier) on top.
Flags games where models disagree beyond DISAGREEMENT_THRESHOLD.

Artifacts saved to models/ directory:
  mlb_ensemble_lr_v{YYYYMMDD_HHMMSS}.joblib
  mlb_ensemble_gbm_v{YYYYMMDD_HHMMSS}.joblib
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from math import exp
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from line_tracker.model import mlb_data
from line_tracker.model import mlb_features

logger = logging.getLogger(__name__)

DISAGREEMENT_THRESHOLD = 0.08  # flag games where models contradict each other
BREAKEVEN_PROB = 110 / 210     # ~0.5238 at -110 juice

_PROJECT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_HOME_MODELS_DIR = Path.home() / ".line_tracker" / "models"
_CLOUD_MODELS_DIR = Path("/mount/src/finproj1/models")

_ENSEMBLE_FEATURE_NAMES = [
    "moneyline_prob",
    "margin_pred",
    "totals_pred",
    "margin_implied_prob",
    "model_disagreement",
]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    ex = exp(x)
    return ex / (1.0 + ex)


def _margin_implied_prob(margin_pred: float) -> float:
    """Convert run-margin prediction to win probability via sigmoid."""
    return _sigmoid(margin_pred / 3.0)


def _models_dir() -> Path:
    return _PROJECT_MODELS_DIR


# ---------------------------------------------------------------------------
# OOF prediction builder
# ---------------------------------------------------------------------------


def build_oof_predictions(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Build out-of-fold predictions from the three base models without data leakage.

    Uses TimeSeriesSplit(n_splits=5) to generate OOF predictions.  Each fold
    trains all three base models on the train split, then predicts on the
    validation split.  Collected outputs form the meta-model training set.

    Parameters
    ----------
    seasons:
        Seasons to include. Defaults to [2022, 2023, 2024, 2025].

    Returns
    -------
    DataFrame with columns:
        moneyline_prob, margin_pred, totals_pred,
        margin_implied_prob, model_disagreement, home_win (target)
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    # ------------------------------------------------------------------
    # Load training data + feature matrix
    # ------------------------------------------------------------------
    print(f"Loading training data for seasons {seasons}...")
    df = mlb_data.load_mlb_training_data(seasons=seasons)
    if df.empty:
        raise RuntimeError("No MLB training data loaded. Check pybaseball connectivity.")

    X, y = mlb_features.build_feature_matrix(df)
    if X.empty:
        raise RuntimeError("Feature matrix is empty after build_feature_matrix.")

    # Defensive numeric cleanup (same as mlb_train.py)
    X = X.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    y_win = y["home_win"].values.astype(int)
    y_margin = y["run_diff"].values.astype(float)
    y_totals = y["total_runs"].values.astype(float)

    n_games = len(X)
    X_arr = X.values  # numpy array for sklearn

    # ------------------------------------------------------------------
    # Out-of-fold loop
    # ------------------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=5)

    oof_moneyline = np.full(n_games, np.nan)
    oof_margin = np.full(n_games, np.nan)
    oof_totals = np.full(n_games, np.nan)

    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X_arr), start=1):
        print(f"  OOF fold {fold_num}/5 — train: {len(train_idx)}, val: {len(val_idx)}")

        X_tr = X_arr[train_idx]
        X_val = X_arr[val_idx]

        # ── Moneyline (LogisticRegression + StandardScaler) ──────────────
        ml_scaler = StandardScaler()
        X_tr_s = ml_scaler.fit_transform(X_tr)
        X_val_s = ml_scaler.transform(X_val)

        ml_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        ml_model.fit(X_tr_s, y_win[train_idx])
        oof_moneyline[val_idx] = ml_model.predict_proba(X_val_s)[:, 1]

        # ── Margin (HistGradientBoostingRegressor, no scaler) ────────────
        from sklearn.ensemble import HistGradientBoostingRegressor

        mg_model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
        )
        mg_model.fit(X.iloc[train_idx].astype(float), y_margin[train_idx])
        oof_margin[val_idx] = mg_model.predict(X.iloc[val_idx].astype(float))

        # ── Totals (HistGradientBoostingRegressor, no scaler) ────────────
        tot_model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
        )
        y_totals_clipped = np.clip(y_totals[train_idx], 3, 15)
        tot_model.fit(X.iloc[train_idx].astype(float), y_totals_clipped)
        oof_totals[val_idx] = tot_model.predict(X.iloc[val_idx].astype(float))

    # Drop rows that remained NaN (first fold's training set has no OOF predictions)
    valid_mask = ~np.isnan(oof_moneyline)

    oof_df = pd.DataFrame({
        "moneyline_prob":     oof_moneyline[valid_mask],
        "margin_pred":        oof_margin[valid_mask],
        "totals_pred":        oof_totals[valid_mask],
        "home_win":           y_win[valid_mask],
    })

    oof_df["margin_implied_prob"] = oof_df["margin_pred"].apply(_margin_implied_prob)
    oof_df["model_disagreement"] = (
        oof_df["moneyline_prob"] - oof_df["margin_implied_prob"]
    ).abs()

    # Reorder columns
    oof_df = oof_df[
        ["moneyline_prob", "margin_pred", "totals_pred",
         "margin_implied_prob", "model_disagreement", "home_win"]
    ]

    print(f"\nOOF DataFrame shape: {oof_df.shape}")
    print("OOF stats:")
    print(oof_df.describe().to_string())
    return oof_df


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------


def train_ensemble(
    seasons: list[int] | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Train meta-ensemble on OOF predictions from the three base models.

    Trains two meta-learners:
        1. LogisticRegression(C=1.0)
        2. GradientBoostingClassifier(n_estimators=200, max_depth=3, lr=0.05)

    Evaluates both with 5-fold StratifiedKFold CV, saves artifacts, and
    prints a comparison table.

    Parameters
    ----------
    seasons:
        Seasons to include. Defaults to [2022, 2023, 2024, 2025].
    models_dir:
        Directory to save model artifacts.

    Returns
    -------
    dict: {"lr": artifact_dict, "gbm": artifact_dict}
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]
    if models_dir is None:
        models_dir = _models_dir()
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Build OOF predictions
    oof_df = build_oof_predictions(seasons=seasons)

    feature_names = _ENSEMBLE_FEATURE_NAMES
    X_ens = oof_df[feature_names].values
    y_ens = oof_df["home_win"].values.astype(int)

    # StratifiedKFold CV for both meta-models
    skf = StratifiedKFold(n_splits=5, shuffle=False)

    # ------------------------------------------------------------------
    # Meta-model 1: LogisticRegression
    # ------------------------------------------------------------------
    lr_scores = []
    for tr, va in skf.split(X_ens, y_ens):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_ens[tr])
        X_va_s = scaler.transform(X_ens[va])
        m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        m.fit(X_tr_s, y_ens[tr])
        lr_scores.append(accuracy_score(y_ens[va], m.predict(X_va_s)))
    lr_cv = float(np.mean(lr_scores))

    # Final fit on all OOF data
    lr_scaler_final = StandardScaler()
    X_ens_scaled = lr_scaler_final.fit_transform(X_ens)
    lr_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_final.fit(X_ens_scaled, y_ens)

    # ------------------------------------------------------------------
    # Meta-model 2: GradientBoostingClassifier
    # ------------------------------------------------------------------
    gbm_scores = []
    for tr, va in skf.split(X_ens, y_ens):
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        m.fit(X_ens[tr], y_ens[tr])
        gbm_scores.append(accuracy_score(y_ens[va], m.predict(X_ens[va])))
    gbm_cv = float(np.mean(gbm_scores))

    # Final fit on all OOF data
    gbm_final = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    gbm_final.fit(X_ens, y_ens)

    # Base moneyline CV score (from OOF accuracy)
    base_ml_cv = float(accuracy_score(
        y_ens,
        (oof_df["moneyline_prob"].values >= 0.5).astype(int)
    ))

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ensemble Training Summary")
    print("=" * 60)
    print(f"{'Model':<20} | {'CV Score':<12} | vs Moneyline Base")
    print("-" * 60)
    print(f"{'Base moneyline':<20} | {base_ml_cv:.1%}{'':>6} | —")
    delta_lr = lr_cv - base_ml_cv
    delta_gbm = gbm_cv - base_ml_cv
    sign_lr = "+" if delta_lr >= 0 else ""
    sign_gbm = "+" if delta_gbm >= 0 else ""
    print(f"{'Ensemble LR':<20} | {lr_cv:.1%}{'':>6} | {sign_lr}{delta_lr:.1%}")
    print(f"{'Ensemble GBM':<20} | {gbm_cv:.1%}{'':>6} | {sign_gbm}{delta_gbm:.1%}")
    print("=" * 60)

    if lr_cv < base_ml_cv:
        print("WARNING: Ensemble LR CV below base moneyline CV. Check OOF features.")
    if gbm_cv < base_ml_cv:
        print("WARNING: Ensemble GBM CV below base moneyline CV. Check OOF features.")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trained_at = datetime.now(timezone.utc).isoformat()

    lr_artifact = {
        "model": lr_final,
        "scaler": lr_scaler_final,
        "metadata": {
            "feature_names": feature_names,
            "cv_score": lr_cv,
            "base_moneyline_cv": base_ml_cv,
            "trained_at": trained_at,
            "disagreement_threshold": DISAGREEMENT_THRESHOLD,
            "seasons": seasons,
            "primary_model": True,  # LR is the primary ensemble model
        },
    }
    lr_path = models_dir / f"mlb_ensemble_lr_v{timestamp}.joblib"
    joblib.dump(lr_artifact, lr_path)
    print(f"Saved LR artifact → {lr_path}")

    gbm_artifact = {
        "model": gbm_final,
        "scaler": None,
        "metadata": {
            "feature_names": feature_names,
            "cv_score": gbm_cv,
            "base_moneyline_cv": base_ml_cv,
            "trained_at": trained_at,
            "disagreement_threshold": DISAGREEMENT_THRESHOLD,
            "seasons": seasons,
            "primary_model": False,  # GBM is reference only; LR is primary
        },
    }
    gbm_path = models_dir / f"mlb_ensemble_gbm_v{timestamp}.joblib"
    joblib.dump(gbm_artifact, gbm_path)
    print(f"Saved GBM artifact → {gbm_path}\n")

    return {
        "lr": lr_artifact,
        "gbm": gbm_artifact,
        "lr_path": str(lr_path),
        "gbm_path": str(gbm_path),
    }


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_ensemble_artifacts(models_dir: Path | None = None) -> dict:
    """Load the most recent LR and GBM ensemble artifacts.

    Parameters
    ----------
    models_dir:
        Directory to search. Defaults to repo-root/models/.

    Returns
    -------
    dict: {"lr": artifact, "gbm": artifact}

    Raises
    ------
    FileNotFoundError:
        If no ensemble artifacts are found. Run train-mlb-ensemble first.
    """
    # If an explicit directory is given, search ONLY that directory.
    # Otherwise fall through the standard search path.
    if models_dir:
        search_dirs = [Path(models_dir)]
    else:
        search_dirs = [_PROJECT_MODELS_DIR, _HOME_MODELS_DIR, _CLOUD_MODELS_DIR]

    result: dict = {}
    for model_type in ("lr", "gbm"):
        pattern = f"mlb_ensemble_{model_type}_*.joblib"
        found = None
        for d in search_dirs:
            candidates = sorted(d.glob(pattern)) if d.exists() else []
            if candidates:
                found = joblib.load(candidates[-1])
                logger.debug("Loaded ensemble %s from %s", model_type, candidates[-1])
                break
        if found is None:
            raise FileNotFoundError(
                f"No MLB ensemble {model_type.upper()} artifact found in {search_dirs}. "
                "Run: python -m line_tracker train-mlb-ensemble"
            )
        result[model_type] = found

    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_ensemble(base_pred_dict: dict) -> dict:
    """Enrich a single game prediction dict with ensemble meta-model outputs.

    Accepts either the short-form keys (moneyline_prob, margin_pred, totals_pred)
    used in testing, or the full predict_mlb_games output keys
    (model_home_win_prob, model_run_diff, model_total_runs).

    Parameters
    ----------
    base_pred_dict:
        Single-game prediction dict from predict_mlb_games() or a test fixture.

    Returns
    -------
    New dict merging input with ensemble fields:
        ensemble_prob, ensemble_lr_prob, margin_implied_prob,
        model_disagreement, flagged, recommended_prob, flag_reason
    """
    # -- Extract base values (support both naming conventions) ----------
    if "moneyline_prob" in base_pred_dict:
        moneyline_prob = float(base_pred_dict["moneyline_prob"])
    else:
        moneyline_prob = float(base_pred_dict.get("model_home_win_prob", 0.5))

    if "margin_pred" in base_pred_dict:
        margin_pred = float(base_pred_dict["margin_pred"])
    else:
        margin_pred = float(base_pred_dict.get("model_run_diff", 0.0))

    if "totals_pred" in base_pred_dict:
        totals_pred = float(base_pred_dict["totals_pred"])
    else:
        totals_pred = float(base_pred_dict.get("model_total_runs", 8.0))

    # -- Derived features -----------------------------------------------
    margin_imp = _margin_implied_prob(margin_pred)
    disagreement = abs(moneyline_prob - margin_imp)
    flagged = disagreement > DISAGREEMENT_THRESHOLD
    flag_reason: str | None = (
        f"disagreement {disagreement:.2f} > threshold {DISAGREEMENT_THRESHOLD}"
        if flagged
        else None
    )

    # -- Load ensemble artifacts ----------------------------------------
    try:
        artifacts = load_ensemble_artifacts()
    except FileNotFoundError:
        # Graceful fallback: return enriched dict without ensemble probs
        logger.warning(
            "Ensemble artifacts not found — skipping ensemble enrichment. "
            "Run: python -m line_tracker train-mlb-ensemble"
        )
        return {
            **base_pred_dict,
            "ensemble_prob": None,
            "ensemble_lr_prob": None,
            "ensemble_gbm_prob": None,
            "margin_implied_prob": round(margin_imp, 4),
            "model_disagreement": round(disagreement, 4),
            "flagged": flagged,
            "recommended_prob": None,
            "flag_reason": flag_reason or "ensemble artifacts not found",
        }

    # -- Build feature vector -------------------------------------------
    feat = np.array([[moneyline_prob, margin_pred, totals_pred, margin_imp, disagreement]])

    # -- LR prediction (PRIMARY — 55.8% CV vs 54.8% for GBM) -----------
    lr_art = artifacts["lr"]
    lr_scaler = lr_art.get("scaler")
    if lr_scaler is not None:
        feat_scaled = lr_scaler.transform(feat)
    else:
        feat_scaled = feat
    ensemble_prob = float(lr_art["model"].predict_proba(feat_scaled)[0][1])

    # -- GBM prediction (reference only) --------------------------------
    gbm_art = artifacts["gbm"]
    ensemble_gbm_prob = float(gbm_art["model"].predict_proba(feat)[0][1])

    # -- Recommended probability (LR as primary) -----------------------
    recommended_prob: float | None = ensemble_prob if not flagged else None

    return {
        **base_pred_dict,
        "ensemble_prob": round(ensemble_prob, 4),        # LR (primary)
        "ensemble_lr_prob": round(ensemble_prob, 4),     # alias for clarity
        "ensemble_gbm_prob": round(ensemble_gbm_prob, 4),  # GBM (reference)
        "margin_implied_prob": round(margin_imp, 4),
        "model_disagreement": round(disagreement, 4),
        "flagged": flagged,
        "recommended_prob": round(recommended_prob, 4) if recommended_prob is not None else None,
        "flag_reason": flag_reason,
    }

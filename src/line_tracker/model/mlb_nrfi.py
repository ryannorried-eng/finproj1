"""
NRFI/YRFI prediction model for MLB first-inning scoring.

Predicts whether at least one run will score in the first inning of a game.

Models
------
mlb_nrfi_gbm  — GradientBoostingClassifier (primary)
mlb_nrfi_lr   — LogisticRegression (secondary / interpretable)

Artifact format: same as existing MLB models
  {"model": fitted, "scaler": StandardScaler | None, "metadata": {...}}
  saved to models/ as mlb_nrfi_{type}_v{YYYYMMDD_HHMMSS}.joblib

Usage
-----
  from line_tracker.model.mlb_nrfi import train_nrfi_model, predict_nrfi
  train_nrfi_model()
  predictions = predict_nrfi()
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from line_tracker.model.mlb_data import (
    CACHE_DIR,
    fetch_batter_season_stats,
    fetch_first_inning_data,
    fetch_game_starters,
    fetch_pitcher_season_stats,
    _fetch_probable_pitchers_mlb,
)
from line_tracker.model.mlb_nrfi_features import (
    NRFI_FEATURE_COLUMNS,
    LEAGUE_AVG_YRFI_RATE,
    _parse_lineup_top3_ids,
    _safe_float,
    build_nrfi_features,
)
from line_tracker.model.mlb_features import PARK_FACTORS

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

_PROJECT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_HOME_MODELS_DIR    = Path.home() / ".line_tracker" / "models"
_CLOUD_MODELS_DIR   = Path("/mount/src/finproj1/models")

EDGE_THRESHOLD   = 0.05
BREAKEVEN_PROB   = 110 / 210   # ~0.5238 at -110 juice


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _search_dirs(models_dir: Path | None) -> list[Path]:
    dirs = []
    if models_dir:
        dirs.append(Path(models_dir))
    dirs.extend([_PROJECT_MODELS_DIR, _HOME_MODELS_DIR, _CLOUD_MODELS_DIR])
    return dirs


def load_nrfi_artifacts(models_dir: Path | None = None) -> dict:
    """Load the most recent NRFI GBM and LR artifacts.

    Returns
    -------
    {"gbm": artifact_dict, "lr": artifact_dict}
    """
    search_dirs = _search_dirs(models_dir)
    artifacts: dict = {}
    for model_key, pattern in [("gbm", "mlb_nrfi_gbm_*.joblib"),
                                 ("lr",  "mlb_nrfi_lr_*.joblib")]:
        for d in search_dirs:
            candidates = sorted(d.glob(pattern)) if d.exists() else []
            if candidates:
                data = joblib.load(candidates[-1])
                artifacts[model_key] = data
                logger.debug("Loaded nrfi_%s from %s", model_key, candidates[-1])
                break
        else:
            raise FileNotFoundError(
                f"No mlb_nrfi_{model_key} model found. "
                "Run: python -m line_tracker train-nrfi"
            )
    return artifacts


def _save_nrfi_artifact(
    model,
    scaler: StandardScaler | None,
    model_key: str,
    cv_roc: float,
    cv_acc: float,
    seasons: list[int],
    n_games: int,
    timestamp: str,
    models_dir: Path,
) -> str:
    filename = f"mlb_nrfi_{model_key}_v{timestamp}.joblib"
    path = models_dir / filename
    metadata = {
        "sport": "baseball_mlb",
        "model_type": f"nrfi_{model_key}",
        "version": "v1_first_inning",
        "seasons_trained": seasons,
        "n_games": n_games,
        "feature_names": NRFI_FEATURE_COLUMNS,
        "cv_roc_auc": cv_roc,
        "cv_accuracy": cv_acc,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "artifact_file": filename,
    }
    joblib.dump({"model": model, "scaler": scaler, "metadata": metadata}, path)
    logger.info("Saved nrfi_%s artifact to %s", model_key, path)
    return str(path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_nrfi_model(
    seasons: list[int] | None = None,
    models_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict:
    """Train NRFI/YRFI GBM and LR models.

    Parameters
    ----------
    seasons:
        Seasons to include.  Defaults to [2022, 2023, 2024, 2025].
    models_dir:
        Directory to save artifacts.  Defaults to repo-root/models/.
    force_refresh:
        Re-fetch cached data.

    Returns
    -------
    {"gbm": path, "lr": path}
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    if models_dir is None:
        models_dir = _PROJECT_MODELS_DIR
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading first-inning data for seasons {seasons}...")
    fi_df = fetch_first_inning_data(seasons=seasons, force_refresh=force_refresh)
    if fi_df.empty:
        raise RuntimeError("No first-inning data loaded.")

    # Load pitcher/batter season stats for feature building
    pitcher_stats: dict[int, pd.DataFrame] = {}
    batter_stats: dict[int, pd.DataFrame] = {}
    for year in seasons:
        try:
            pitcher_stats[year] = fetch_pitcher_season_stats(year)
        except Exception as exc:
            logger.warning("Could not load pitcher stats for %d: %s", year, exc)
        try:
            batter_stats[year] = fetch_batter_season_stats(year)
        except Exception as exc:
            logger.warning("Could not load batter stats for %d: %s", year, exc)

    print("Building NRFI features...")
    feat_df = build_nrfi_features(
        fi_df,
        pitcher_stats=pitcher_stats,
        batter_stats=batter_stats,
    )

    feat_df = feat_df.dropna(subset=["yrfi"])
    X = feat_df[NRFI_FEATURE_COLUMNS].copy().astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = feat_df["yrfi"].astype(int)

    n_games = len(X)
    yrfi_rate = float(y.mean())
    print(f"Training on {n_games} games | YRFI rate: {yrfi_rate:.3f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- GBM ---
    gbm_roc_scores, gbm_acc_scores = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_val)[:, 1]
        gbm_roc_scores.append(roc_auc_score(y_val, probs))
        gbm_acc_scores.append(accuracy_score(y_val, m.predict(X_val)))

    gbm_cv_roc = float(np.mean(gbm_roc_scores))
    gbm_cv_acc = float(np.mean(gbm_acc_scores))

    gbm_final = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm_final.fit(X, y)

    # --- LR ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_roc_scores, lr_acc_scores = [], []
    X_np = X.values
    y_np = y.values
    for train_idx, val_idx in skf.split(X_np, y_np):
        X_tr = scaler.transform(X_np[train_idx])
        X_val = scaler.transform(X_np[val_idx])
        y_tr, y_val = y_np[train_idx], y_np[val_idx]

        m = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_val)[:, 1]
        lr_roc_scores.append(roc_auc_score(y_val, probs))
        lr_acc_scores.append(accuracy_score(y_val, m.predict(X_val)))

    lr_cv_roc = float(np.mean(lr_roc_scores))
    lr_cv_acc = float(np.mean(lr_acc_scores))

    lr_final = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    lr_final.fit(X_scaled, y)

    # --- Save ---
    gbm_path = _save_nrfi_artifact(
        gbm_final, None, "gbm", gbm_cv_roc, gbm_cv_acc,
        seasons, n_games, timestamp, models_dir,
    )
    lr_path = _save_nrfi_artifact(
        lr_final, scaler, "lr", lr_cv_roc, lr_cv_acc,
        seasons, n_games, timestamp, models_dir,
    )

    # --- Summary ---
    naive_acc = max(yrfi_rate, 1 - yrfi_rate)
    print("\n" + "=" * 64)
    print("NRFI/YRFI Model Training Summary")
    print("=" * 64)
    print(f"{'Model':<16} | {'CV ROC-AUC':>10} | {'CV Accuracy':>11} | YRFI Rate")
    print("-" * 64)
    print(f"{'GBM':<16} | {gbm_cv_roc:>10.3f} | {gbm_cv_acc:>10.1%} | {yrfi_rate:.1%}")
    print(f"{'LR':<16} | {lr_cv_roc:>10.3f} | {lr_cv_acc:>10.1%} | {yrfi_rate:.1%}")
    print(f"{'Naive baseline':<16} | {'0.500':>10} | {naive_acc:>10.1%} | {yrfi_rate:.1%}")
    print("=" * 64 + "\n")

    return {"gbm": gbm_path, "lr": lr_path}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _build_prediction_features(
    game_pk: int,
    home_sp_id: int | None,
    away_sp_id: int | None,
    home_team: str,
    away_team: str,
    home_lineup_json: str | None,
    away_lineup_json: str | None,
    pitcher_stats_df: pd.DataFrame,
    batter_stats_df: pd.DataFrame,
    fi_history: pd.DataFrame,
    game_date: date,
) -> dict:
    """Build a single-game feature vector for NRFI prediction.

    Returns dict with keys = NRFI_FEATURE_COLUMNS.
    """
    feat: dict[str, float] = {}

    # SP season stats
    for side, sp_id in [("home", home_sp_id), ("away", away_sp_id)]:
        if sp_id is not None and not pitcher_stats_df.empty:
            try:
                row = pitcher_stats_df.loc[int(sp_id)]
                feat[f"{side}_sp_k9"]   = _safe_float(row.get("k9"),   8.5)
                feat[f"{side}_sp_bb9"]  = _safe_float(row.get("bb9"),  3.2)
                feat[f"{side}_sp_whip"] = _safe_float(row.get("whip"), 1.30)
                feat[f"{side}_sp_fi_era"] = _safe_float(row.get("era"), 4.20)
            except (KeyError, TypeError):
                feat[f"{side}_sp_k9"]   = 8.5
                feat[f"{side}_sp_bb9"]  = 3.2
                feat[f"{side}_sp_whip"] = 1.30
                feat[f"{side}_sp_fi_era"] = 4.20
        else:
            feat[f"{side}_sp_k9"]   = 8.5
            feat[f"{side}_sp_bb9"]  = 3.2
            feat[f"{side}_sp_whip"] = 1.30
            feat[f"{side}_sp_fi_era"] = 4.20

    # SP first-inning YRFI rate from history
    for side, sp_id, runs_col in [
        ("home", home_sp_id, "first_inning_away_runs"),
        ("away", away_sp_id, "first_inning_home_runs"),
    ]:
        rate = LEAGUE_AVG_YRFI_RATE
        fi_era = feat[f"{side}_sp_fi_era"]
        if sp_id is not None and not fi_history.empty:
            sp_games = fi_history[fi_history[f"{side}_sp_id"] == sp_id].tail(10)
            if len(sp_games) >= 1:
                rate = float(sp_games["yrfi"].mean())
            if len(sp_games) >= 1:
                fi_era = float(sp_games[runs_col].mean() * 9)
        feat[f"{side}_sp_fi_yrfi_rate_r10"] = rate
        feat[f"{side}_sp_fi_era"] = fi_era

    # Team rolling YRFI rate
    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        team = home_team if side == "home" else away_team
        if not fi_history.empty:
            team_games = fi_history[fi_history[team_col] == team].tail(15)
            feat[f"{side}_team_yrfi_rate_r15"] = float(team_games["yrfi"].mean()) if len(team_games) > 0 else LEAGUE_AVG_YRFI_RATE
        else:
            feat[f"{side}_team_yrfi_rate_r15"] = LEAGUE_AVG_YRFI_RATE

    # Batter OBP features
    for side, lineup_json in [("home", home_lineup_json), ("away", away_lineup_json)]:
        top3_ids = _parse_lineup_top3_ids(lineup_json)
        obps = []
        for bid in top3_ids:
            if not batter_stats_df.empty:
                try:
                    obp = _safe_float(batter_stats_df.loc[int(bid)].get("obp"))
                    if not np.isnan(obp):
                        obps.append(obp)
                except (KeyError, TypeError):
                    pass
        avg_obp = float(np.mean(obps)) if obps else 0.320
        feat[f"{side}_top3_obp_r15"]    = avg_obp
        feat[f"{side}_team_obp_r15"]    = avg_obp
        feat[f"{side}_top3_k_rate_r15"] = 0.22   # league average (no game logs in predict)

    # Park factor
    feat["park_factor_runs"] = PARK_FACTORS.get(str(home_team), {}).get("runs", 100) / 100.0

    # Weather defaults
    feat["temp_f"]          = 72.0
    feat["wind_out_factor"] = 0.0
    feat["is_day_game"]     = 0

    # Ensure all feature columns present
    for col in NRFI_FEATURE_COLUMNS:
        if col not in feat:
            feat[col] = 0.0

    return feat


def _implied_prob(american_odds: float | None) -> float | None:
    if american_odds is None or np.isnan(american_odds):
        return None
    if american_odds < 0:
        return -american_odds / (-american_odds + 100)
    return 100 / (american_odds + 100)


def _fetch_nrfi_odds(game_pks: list[int]) -> dict[int, dict]:
    """Attempt to fetch NRFI/YRFI market odds from The Odds API.

    Returns dict: game_pk → {"yrfi_odds": float|None, "nrfi_odds": float|None}.
    If unavailable, returns empty dict (no NRFI/YRFI market available).
    """
    # NRFI/YRFI is an alternate market not always available via The Odds API.
    # Return empty to skip edge calc — market_yrfi_prob will be None.
    return {}


# ---------------------------------------------------------------------------
# predict_nrfi
# ---------------------------------------------------------------------------


def predict_nrfi(
    game_date: date | None = None,
    use_odds: bool = True,
    models_dir: Path | None = None,
) -> list[dict]:
    """Generate NRFI/YRFI predictions for all games on a given date.

    Parameters
    ----------
    game_date:
        Date to predict.  Defaults to today.
    use_odds:
        If True, attempt to fetch NRFI/YRFI market odds for edge calculation.
        Gracefully handles unavailable markets (sets market_yrfi_prob=None).
    models_dir:
        Override model search directory.

    Returns
    -------
    list of dicts, one per game:
        game, home_sp, away_sp,
        yrfi_prob, nrfi_prob,
        market_yrfi_prob (None if unavailable),
        yrfi_edge, nrfi_edge,
        bet ('YRFI' | 'NRFI' | None),
        edge (float | None),
        confidence ('high' | 'medium' | 'low' | None),
        home_sp_fi_yrfi_rate, away_sp_fi_yrfi_rate,
        home_top3_obp, away_top3_obp,
    """
    if game_date is None:
        game_date = date.today()

    # Load models
    try:
        artifacts = load_nrfi_artifacts(models_dir)
    except FileNotFoundError as exc:
        logger.warning("NRFI model not found: %s", exc)
        return []

    gbm_model = artifacts["gbm"]["model"]
    lr_model  = artifacts["lr"]["model"]
    lr_scaler = artifacts["lr"]["scaler"]

    # Get today's games and probable pitchers
    try:
        probable = _fetch_probable_pitchers_mlb(game_date)
    except Exception as exc:
        logger.warning("Could not fetch probable pitchers for %s: %s", game_date, exc)
        return []

    if probable.empty:
        return []

    # Load current-season pitcher / batter stats (prefer most recent year)
    current_year = game_date.year
    training_year = current_year - 1  # use prior season stats as training proxy

    try:
        pitcher_stats_df = fetch_pitcher_season_stats(training_year)
    except Exception:
        pitcher_stats_df = pd.DataFrame()

    try:
        batter_stats_df = fetch_batter_season_stats(training_year)
    except Exception:
        batter_stats_df = pd.DataFrame()

    # Load first-inning history for rolling rates
    try:
        fi_history = fetch_first_inning_data(seasons=[training_year])
    except Exception:
        fi_history = pd.DataFrame()

    # Fetch NRFI market odds if requested
    odds_map: dict[int, dict] = {}
    if use_odds:
        try:
            game_pks = probable["game_pk"].dropna().astype(int).tolist()
            odds_map = _fetch_nrfi_odds(game_pks)
        except Exception:
            pass

    # Get today's game starters (for lineup data)
    try:
        starters_today = fetch_game_starters(current_year, force_refresh=False)
    except Exception:
        starters_today = pd.DataFrame()

    results: list[dict] = []

    for _, game in probable.iterrows():
        try:
            game_pk      = int(game["game_pk"]) if pd.notna(game.get("game_pk")) else None
            home_team    = str(game["home_team"])
            away_team    = str(game["away_team"])
            home_sp_id   = game.get("home_pitcher_id")
            away_sp_id   = game.get("away_pitcher_id")
            home_sp_name = str(game.get("home_pitcher_name") or "TBD")
            away_sp_name = str(game.get("away_pitcher_name") or "TBD")

            # Try to get lineup from starters cache
            home_lineup_json: str | None = None
            away_lineup_json: str | None = None
            if not starters_today.empty and game_pk is not None:
                match = starters_today[starters_today["game_pk"] == game_pk]
                if not match.empty:
                    home_lineup_json = match.iloc[0].get("home_lineup")
                    away_lineup_json = match.iloc[0].get("away_lineup")

            feat = _build_prediction_features(
                game_pk=game_pk or 0,
                home_sp_id=int(home_sp_id) if pd.notna(home_sp_id) else None,
                away_sp_id=int(away_sp_id) if pd.notna(away_sp_id) else None,
                home_team=home_team,
                away_team=away_team,
                home_lineup_json=home_lineup_json,
                away_lineup_json=away_lineup_json,
                pitcher_stats_df=pitcher_stats_df,
                batter_stats_df=batter_stats_df,
                fi_history=fi_history,
                game_date=game_date,
            )

            X_row = np.array([[feat[c] for c in NRFI_FEATURE_COLUMNS]], dtype=float)
            X_row = np.nan_to_num(X_row, nan=0.0)

            gbm_prob = float(gbm_model.predict_proba(X_row)[0, 1])
            if lr_scaler is not None:
                X_lr = lr_scaler.transform(X_row)
            else:
                X_lr = X_row
            lr_prob = float(lr_model.predict_proba(X_lr)[0, 1])

            # Ensemble: weighted average (GBM primary)
            yrfi_prob = 0.65 * gbm_prob + 0.35 * lr_prob
            nrfi_prob = 1.0 - yrfi_prob

            # Market odds
            market_yrfi_prob: float | None = None
            yrfi_edge: float | None = None
            nrfi_edge: float | None = None
            bet: str | None = None
            edge: float | None = None
            confidence: str | None = None

            if game_pk and game_pk in odds_map:
                mkt = odds_map[game_pk]
                yrfi_implied = _implied_prob(mkt.get("yrfi_odds"))
                nrfi_implied = _implied_prob(mkt.get("nrfi_odds"))
                if yrfi_implied:
                    market_yrfi_prob = yrfi_implied
                    yrfi_edge = yrfi_prob - yrfi_implied
                    nrfi_edge = nrfi_prob - (nrfi_implied or (1 - yrfi_implied))

                    if yrfi_edge is not None and nrfi_edge is not None:
                        if yrfi_edge >= EDGE_THRESHOLD:
                            bet  = "YRFI"
                            edge = yrfi_edge
                        elif nrfi_edge >= EDGE_THRESHOLD:
                            bet  = "NRFI"
                            edge = nrfi_edge

                    if edge is not None:
                        if edge >= 0.08:
                            confidence = "high"
                        elif edge >= EDGE_THRESHOLD:
                            confidence = "medium"
                        else:
                            confidence = "low"

            results.append({
                "game":                   f"{away_team} @ {home_team}",
                "home_team":              home_team,
                "away_team":              away_team,
                "game_pk":                game_pk,
                "home_sp":                home_sp_name,
                "away_sp":                away_sp_name,
                "yrfi_prob":              round(yrfi_prob, 4),
                "nrfi_prob":              round(nrfi_prob, 4),
                "market_yrfi_prob":       round(market_yrfi_prob, 4) if market_yrfi_prob else None,
                "yrfi_edge":              round(yrfi_edge, 4) if yrfi_edge is not None else None,
                "nrfi_edge":              round(nrfi_edge, 4) if nrfi_edge is not None else None,
                "bet":                    bet,
                "edge":                   round(edge, 4) if edge is not None else None,
                "confidence":             confidence,
                "home_sp_fi_yrfi_rate":   round(feat["home_sp_fi_yrfi_rate_r10"], 3),
                "away_sp_fi_yrfi_rate":   round(feat["away_sp_fi_yrfi_rate_r10"], 3),
                "home_top3_obp":          round(feat["home_top3_obp_r15"], 3),
                "away_top3_obp":          round(feat["away_top3_obp_r15"], 3),
            })

        except Exception as exc:
            logger.warning("NRFI prediction failed for game: %s", exc)
            continue

    return results

"""NRFI/YRFI first-inning probability estimates for MLB games."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EDGE_THRESHOLD = 0.05
BREAKEVEN_PROB = 0.524  # win rate needed to break even at -110 odds

_LEAGUE_AVG_ERA = 4.20
_BASE_PER_TEAM_YRFI = 0.30  # ~51% combined YRFI at league-avg ERA
_ERA_SCALE = 0.025  # probability shift per ERA point above/below league avg

_PROJECT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_HOME_MODELS_DIR = Path.home() / ".line_tracker" / "models"
_CLOUD_MODELS_DIR = Path("/mount/src/finproj1/models")


def _per_team_yrfi(era: float | None) -> float:
    if era is None:
        return _BASE_PER_TEAM_YRFI
    return max(0.10, min(0.60, _BASE_PER_TEAM_YRFI + (era - _LEAGUE_AVG_ERA) * _ERA_SCALE))


def _sp_fi_yrfi_rate(fi_history, sp_id) -> float | None:
    """Return a starter's last-10-start first-inning YRFI rate, or None."""
    if fi_history is None or fi_history.empty or sp_id is None:
        return None
    try:
        sp_id_int = int(sp_id)
    except (TypeError, ValueError):
        return None
    mask = (
        (fi_history["home_sp_id"] == sp_id_int)
        | (fi_history["away_sp_id"] == sp_id_int)
    )
    sp_games = fi_history[mask].sort_values("date").tail(10)
    if sp_games.empty:
        return None
    return float(sp_games["yrfi"].mean())


def load_nrfi_artifacts(models_dir: Path | None = None) -> dict:
    """Load the most recent NRFI GBM and LR model artifacts.

    Returns
    -------
    dict: {"gbm": artifact_dict, "lr": artifact_dict}
    Each artifact_dict has keys: model, scaler, metadata.
    """
    import joblib

    search_dirs: list[Path] = []
    if models_dir:
        search_dirs.append(Path(models_dir))
    search_dirs.extend([_PROJECT_MODELS_DIR, _HOME_MODELS_DIR, _CLOUD_MODELS_DIR])

    artifacts: dict = {}
    for model_type in ("gbm", "lr"):
        pattern = f"mlb_nrfi_{model_type}_*.joblib"
        for d in search_dirs:
            if not d.exists():
                continue
            candidates = sorted(d.glob(pattern))
            if candidates:
                artifacts[model_type] = joblib.load(candidates[-1])
                logger.debug("Loaded NRFI %s from %s", model_type, candidates[-1])
                break
        else:
            raise FileNotFoundError(
                f"No NRFI {model_type} model found in {search_dirs}"
            )

    return artifacts


def _build_prediction_features(
    game_pk: int,
    home_sp_id,
    away_sp_id,
    home_team: str,
    away_team: str,
    home_lineup_json,
    away_lineup_json,
    pitcher_stats_df,
    batter_stats_df,
    fi_history,
    game_date,
) -> dict:
    """Build the 22-feature dict for a single game NRFI prediction.

    Weather (temp_f, wind_out_factor) and is_day_game are set to defaults
    here and should be overridden by the caller with actual values.
    """
    import pandas as pd
    from line_tracker.model.mlb_nrfi_features import (
        LEAGUE_AVG_BB9,
        LEAGUE_AVG_ERA,
        LEAGUE_AVG_K9,
        LEAGUE_AVG_OBP,
        LEAGUE_AVG_WHIP,
        LEAGUE_AVG_YRFI_RATE,
        _safe_float,
    )
    from line_tracker.model.mlb_features import PARK_FACTORS

    # ------------------------------------------------------------------
    # SP rolling first-inning stats from fi_history
    # ------------------------------------------------------------------
    def _sp_fi_stats(sp_id, side: str):
        if fi_history is None or fi_history.empty or sp_id is None:
            return LEAGUE_AVG_ERA, LEAGUE_AVG_YRFI_RATE
        try:
            sp_id_int = int(sp_id)
        except (TypeError, ValueError):
            return LEAGUE_AVG_ERA, LEAGUE_AVG_YRFI_RATE

        if side == "home":
            mask = fi_history["home_sp_id"] == sp_id_int
            runs_col = "first_inning_away_runs"
        else:
            mask = fi_history["away_sp_id"] == sp_id_int
            runs_col = "first_inning_home_runs"

        sp_games = fi_history[mask].sort_values("date").tail(10)
        if sp_games.empty:
            return LEAGUE_AVG_ERA, LEAGUE_AVG_YRFI_RATE

        if runs_col in sp_games.columns:
            fi_era = float(pd.to_numeric(sp_games[runs_col], errors="coerce").mean() * 9)
            if np.isnan(fi_era):
                fi_era = LEAGUE_AVG_ERA
        else:
            fi_era = LEAGUE_AVG_ERA

        fi_yrfi = float(sp_games["yrfi"].mean()) if "yrfi" in sp_games.columns else LEAGUE_AVG_YRFI_RATE
        return fi_era, fi_yrfi

    home_fi_era, home_fi_yrfi = _sp_fi_stats(home_sp_id, "home")
    away_fi_era, away_fi_yrfi = _sp_fi_stats(away_sp_id, "away")

    # ------------------------------------------------------------------
    # SP season stats (ERA, K9, BB9, WHIP) from pitcher_stats_df
    # ------------------------------------------------------------------
    def _get_sp_stat(sp_id, col: str, default: float) -> float:
        if pitcher_stats_df is None or pitcher_stats_df.empty or sp_id is None:
            return default
        try:
            sp_id_int = int(sp_id)
        except (TypeError, ValueError):
            return default
        if sp_id_int not in pitcher_stats_df.index:
            return default
        val = _safe_float(pitcher_stats_df.loc[sp_id_int].get(col), default)
        return val if not np.isnan(val) else default

    # ------------------------------------------------------------------
    # Team rolling YRFI rates from fi_history
    # ------------------------------------------------------------------
    def _team_yrfi_rate(team: str, side: str) -> float:
        if fi_history is None or fi_history.empty:
            return LEAGUE_AVG_YRFI_RATE
        col = f"{side}_team"
        if col not in fi_history.columns:
            return LEAGUE_AVG_YRFI_RATE
        mask = fi_history[col] == team
        team_games = fi_history[mask].sort_values("date").tail(15)
        if team_games.empty:
            return LEAGUE_AVG_YRFI_RATE
        return float(team_games["yrfi"].mean())

    # ------------------------------------------------------------------
    # Park factor
    # ------------------------------------------------------------------
    pf = PARK_FACTORS.get(home_team, {"runs": 100, "hr": 100})
    park_factor_runs = pf["runs"] / 100.0

    return {
        "home_sp_fi_era":            home_fi_era,
        "away_sp_fi_era":            away_fi_era,
        "home_sp_k9":                _get_sp_stat(home_sp_id, "k9",   LEAGUE_AVG_K9),
        "away_sp_k9":                _get_sp_stat(away_sp_id, "k9",   LEAGUE_AVG_K9),
        "home_sp_bb9":               _get_sp_stat(home_sp_id, "bb9",  LEAGUE_AVG_BB9),
        "away_sp_bb9":               _get_sp_stat(away_sp_id, "bb9",  LEAGUE_AVG_BB9),
        "home_sp_whip":              _get_sp_stat(home_sp_id, "whip", LEAGUE_AVG_WHIP),
        "away_sp_whip":              _get_sp_stat(away_sp_id, "whip", LEAGUE_AVG_WHIP),
        "home_sp_fi_yrfi_rate_r10":  home_fi_yrfi,
        "away_sp_fi_yrfi_rate_r10":  away_fi_yrfi,
        # Lineup=None → league average OBP / K-rate
        "home_top3_obp_r15":         LEAGUE_AVG_OBP,
        "away_top3_obp_r15":         LEAGUE_AVG_OBP,
        "home_top3_k_rate_r15":      0.22,
        "away_top3_k_rate_r15":      0.22,
        "home_team_obp_r15":         LEAGUE_AVG_OBP,
        "away_team_obp_r15":         LEAGUE_AVG_OBP,
        "park_factor_runs":          park_factor_runs,
        # Weather defaults — caller should override with actual values
        "temp_f":                    72.0,
        "wind_out_factor":           0.0,
        "is_day_game":               0,
        "home_team_yrfi_rate_r15":   _team_yrfi_rate(home_team, "home"),
        "away_team_yrfi_rate_r15":   _team_yrfi_rate(away_team, "away"),
    }


def predict_nrfi(
    target_date: date | None = None,
    fi_history=None,  # pre-loaded fi_history; if None, loads internally (CLI usage)
) -> list[dict]:
    """Return NRFI/YRFI probability estimates for each game on target_date.

    Loads first-inning history for the prior and current season so that
    per-SP rolling YRFI rates reflect 2026 actual starts when available,
    falling back to 2025 data for pitchers with fewer than one 2026 start.

    Keys in each returned dict:
        game_pk, away_team, home_team, away_team_br, home_team_br,
        away_pitcher, home_pitcher, away_pitcher_era, home_pitcher_era,
        away_sp_fi_yrfi_rate, home_sp_fi_yrfi_rate,
        yrfi_prob, nrfi_prob, commence_time,
        gbm_prob, lr_prob, model_type
    """
    import pandas as pd
    from line_tracker.model.mlb_nrfi_features import NRFI_FEATURE_COLUMNS

    if target_date is None:
        target_date = date.today()
    current_year = target_date.year

    # Load first-inning history only when not pre-supplied by caller.
    if fi_history is None:
        try:
            from line_tracker.model.mlb_data import fetch_first_inning_data
            df = fetch_first_inning_data(
                seasons=[current_year - 1, current_year],
                force_refresh=False,
            )
            if df is not None and not df.empty:
                fi_history = df
        except Exception as exc:
            logger.debug("Could not load first-inning history: %s", exc)

    # ------------------------------------------------------------------
    # Step 1 — Load model artifacts
    # ------------------------------------------------------------------
    try:
        artifacts = load_nrfi_artifacts()
        gbm_model = artifacts["gbm"]["model"]
        lr_model = artifacts["lr"]["model"]
        lr_scaler = artifacts["lr"]["scaler"]
        use_full_model = True
    except FileNotFoundError:
        use_full_model = False
        gbm_model = lr_model = lr_scaler = None

    # ------------------------------------------------------------------
    # Step 2 — Load pitcher stats once
    # ------------------------------------------------------------------
    try:
        from line_tracker.model.mlb_data import fetch_pitcher_season_stats
        pitcher_stats_df = fetch_pitcher_season_stats(current_year - 1)
    except Exception:
        pitcher_stats_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Step 3 — Load batter stats once
    # ------------------------------------------------------------------
    try:
        from line_tracker.model.mlb_data import fetch_batter_season_stats
        batter_stats_df = fetch_batter_season_stats(current_year - 1)
    except Exception:
        batter_stats_df = pd.DataFrame()

    try:
        from line_tracker.model.mlb_predict import predict_mlb_games
        preds = predict_mlb_games(target_date=target_date)
    except Exception:
        return []

    results = []
    for p in preds:
        away_era = p.get("away_pitcher_era")
        home_era = p.get("home_pitcher_era")
        home_pid = p.get("home_pitcher_id")
        away_pid = p.get("away_pitcher_id")
        home_team = p.get("home_team_br") or p.get("home_team", "")
        away_team = p.get("away_team_br") or p.get("away_team", "")
        game_pk = p.get("game_id", 0)

        home_fi_rate = _sp_fi_yrfi_rate(fi_history, home_pid)
        away_fi_rate = _sp_fi_yrfi_rate(fi_history, away_pid)

        gbm_prob: float | None = None
        lr_prob: float | None = None

        if use_full_model:
            try:
                # ------------------------------------------------------------------
                # Step 4 — Build feature vector
                # ------------------------------------------------------------------
                feat = _build_prediction_features(
                    game_pk=game_pk or 0,
                    home_sp_id=home_pid,
                    away_sp_id=away_pid,
                    home_team=home_team,
                    away_team=away_team,
                    home_lineup_json=None,
                    away_lineup_json=None,
                    pitcher_stats_df=pitcher_stats_df,
                    batter_stats_df=batter_stats_df,
                    fi_history=fi_history,
                    game_date=target_date,
                )

                # Override weather with actual forecast values
                feat["temp_f"] = float(p.get("temp_f") or 72.0)
                feat["wind_out_factor"] = float(p.get("wind_out_factor") or 0.0)

                # Derive is_day_game from commence_time (UTC hour < 17 → day game)
                ct = p.get("commence_time", "")
                game_time_hour = 19  # default: night game
                if ct:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(ct)
                        game_time_hour = dt.hour
                    except Exception:
                        pass
                feat["is_day_game"] = 1 if game_time_hour < 17 else 0

                # ------------------------------------------------------------------
                # Step 5 — Score with GBM + LR ensemble
                # ------------------------------------------------------------------
                X_row = np.array(
                    [[feat[c] for c in NRFI_FEATURE_COLUMNS]], dtype=float
                )
                X_row = np.nan_to_num(X_row, nan=0.0)

                gbm_prob = float(gbm_model.predict_proba(X_row)[0, 1])

                # Use GBM only — LR artifact has sklearn version mismatch
                try:
                    if lr_scaler is not None:
                        X_lr = lr_scaler.transform(X_row)
                    else:
                        X_lr = X_row
                    lr_prob = float(lr_model.predict_proba(X_lr)[0, 1])
                    combined_yrfi = round(0.65 * gbm_prob + 0.35 * lr_prob, 4)
                except Exception:
                    lr_prob = None
                    combined_yrfi = round(float(gbm_prob), 4)
                model_type = "gbm_lr_ensemble"

            except Exception as exc:
                logger.warning("Model scoring failed for %s@%s: %s", away_team, home_team, exc)
                use_full_model_this_game = False
                gbm_prob = lr_prob = None
                model_type = "era_formula"
                # Fall through to ERA formula below
            else:
                use_full_model_this_game = True
        else:
            use_full_model_this_game = False
            model_type = "era_formula"

        # ------------------------------------------------------------------
        # Step 6 — ERA formula fallback
        # ------------------------------------------------------------------
        if not use_full_model or (gbm_prob is None):
            era_home = _per_team_yrfi(home_era)
            era_away = _per_team_yrfi(away_era)
            home_yrfi = (0.25 * home_fi_rate + 0.75 * era_home) if home_fi_rate is not None else era_home
            away_yrfi = (0.25 * away_fi_rate + 0.75 * era_away) if away_fi_rate is not None else era_away
            combined_yrfi = round(1.0 - (1.0 - away_yrfi) * (1.0 - home_yrfi), 4)
            model_type = "era_formula"

        results.append({
            "game_pk": game_pk,
            "away_team": p.get("away_team", ""),
            "home_team": p.get("home_team", ""),
            "away_team_br": p.get("away_team_br", ""),
            "home_team_br": p.get("home_team_br", ""),
            "away_pitcher": p.get("away_pitcher") or "TBD",
            "home_pitcher": p.get("home_pitcher") or "TBD",
            "away_pitcher_era": away_era,
            "home_pitcher_era": home_era,
            "away_sp_fi_yrfi_rate": round(away_fi_rate, 3) if away_fi_rate is not None else None,
            "home_sp_fi_yrfi_rate": round(home_fi_rate, 3) if home_fi_rate is not None else None,
            "yrfi_prob": combined_yrfi,
            "nrfi_prob": round(1.0 - combined_yrfi, 4),
            "commence_time": p.get("commence_time", ""),
            "gbm_prob": round(gbm_prob, 4) if gbm_prob is not None else None,
            "lr_prob": round(lr_prob, 4) if lr_prob is not None else None,
            "model_type": model_type,
        })
    return results

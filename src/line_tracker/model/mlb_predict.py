"""
MLB model predictions for upcoming games.
Loads the most recent mlb_moneyline, mlb_margin, mlb_totals artifacts.
Fetches today's MLB odds from The Odds API (sport key: baseball_mlb).
Builds features from 2025 season-end trailing stats as baseline.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from line_tracker.model.mlb_features import FEATURE_COLUMNS, PARK_FACTORS

logger = logging.getLogger(__name__)

# Odds API full name → Baseball Reference abbreviation
ODDS_TO_BR: dict[str, str] = {
    "Atlanta Braves": "ATL", "Arizona Diamondbacks": "ARI",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Athletics": "OAK", "Sacramento Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFG", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}

_PROJECT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
_HOME_MODELS_DIR = Path.home() / ".line_tracker" / "models"
_CLOUD_MODELS_DIR = Path("/mount/src/finproj1/models")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_mlb_artifacts(models_dir: Path | None = None) -> dict:
    """Load the most recent MLB model artifacts for all three model types.

    Returns
    -------
    dict: {"moneyline": artifact_dict, "margin": artifact_dict, "totals": artifact_dict}
    """
    search_dirs = []
    if models_dir:
        search_dirs.append(Path(models_dir))
    search_dirs.extend([_PROJECT_MODELS_DIR, _HOME_MODELS_DIR, _CLOUD_MODELS_DIR])

    artifacts: dict = {}
    for model_type in ("moneyline", "margin", "totals"):
        pattern = f"mlb_{model_type}_*.joblib"
        for d in search_dirs:
            candidates = sorted(d.glob(pattern)) if d.exists() else []
            if candidates:
                data = joblib.load(candidates[-1])
                artifacts[model_type] = data
                logger.debug("Loaded %s from %s", model_type, candidates[-1])
                break
        else:
            msg = (
                f"No MLB {model_type} model found. "
                f"Run: python -m line_tracker train-mlb-model"
            )
            raise FileNotFoundError(msg)

    return artifacts


# ---------------------------------------------------------------------------
# Trailing stats
# ---------------------------------------------------------------------------


def get_trailing_stats(season: int = 2025) -> dict[str, dict]:
    """Compute season-end trailing stats for each team.

    Uses the last 15 games (R15) and last 10 games (R10) of the season.
    These stats are used for Opening Day / early season predictions before
    current-season rolling stats accumulate.

    Returns
    -------
    dict: {team_abbr: {runs_scored_r15, runs_allowed_r15, run_diff_r15,
                        k_rate_r15, bb_rate_r15, run_diff_r10}}
    """
    from line_tracker.model.mlb_data import CACHE_DIR, fetch_team_game_logs

    cache_path = CACHE_DIR / f"trailing_stats_{season}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        return cached.set_index("team").to_dict("index")

    logs = fetch_team_game_logs(season)
    if logs.empty:
        logger.warning("No game logs for season %d; using league-average defaults", season)
        return _league_avg_trailing_stats()

    result: dict[str, dict] = {}

    for team, group in logs.groupby("team"):
        group = group.sort_values("date").copy()

        last15 = group.tail(15)
        last10 = group.tail(10)

        rs15 = float(last15["runs_scored"].mean()) if last15["runs_scored"].notna().any() else 4.5
        ra15 = float(last15["runs_allowed"].mean()) if last15["runs_allowed"].notna().any() else 4.5

        denom15 = last15["hits"] + last15["walks"] + last15["strikeouts"]
        safe_denom15 = denom15.replace(0, np.nan)
        k15 = float((last15["strikeouts"] / safe_denom15).mean()) if safe_denom15.notna().any() else 0.20
        bb15 = float((last15["walks"] / safe_denom15).mean()) if safe_denom15.notna().any() else 0.09

        rd15 = rs15 - ra15

        rd10 = float(
            (last10["runs_scored"] - last10["runs_allowed"]).mean()
        ) if len(last10) > 0 else 0.0

        result[str(team)] = {
            "runs_scored_r15": rs15,
            "runs_allowed_r15": ra15,
            "run_diff_r15": rd15,
            "k_rate_r15": k15,
            "bb_rate_r15": bb15,
            "run_diff_r10": rd10,
        }

    # Cache as parquet
    rows = [{"team": k, **v} for k, v in result.items()]
    pd.DataFrame(rows).to_parquet(cache_path, index=False)

    return result


def _league_avg_trailing_stats() -> dict[str, dict]:
    """Return league-average trailing stats as a fallback."""
    from line_tracker.model.mlb_data import MLB_TEAMS
    avg = {
        "runs_scored_r15": 4.5,
        "runs_allowed_r15": 4.5,
        "run_diff_r15": 0.0,
        "k_rate_r15": 0.20,
        "bb_rate_r15": 0.09,
        "run_diff_r10": 0.0,
    }
    return {team: avg.copy() for team in MLB_TEAMS}


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------


def build_prediction_features(
    home_team_br: str,
    away_team_br: str,
    trailing_stats: dict[str, dict],
    home_days_rest: int = 1,
    away_days_rest: int = 1,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for a game prediction.

    Parameters
    ----------
    home_team_br:
        Home team BR abbreviation (e.g. "LAD").
    away_team_br:
        Away team BR abbreviation (e.g. "SFG").
    trailing_stats:
        Dict of {team_abbr: {feature: value}} from get_trailing_stats().
    home_days_rest:
        Days since home team's last game (default: 1, capped at 7).
    away_days_rest:
        Days since away team's last game (default: 1, capped at 7).

    Returns
    -------
    Single-row DataFrame with FEATURE_COLUMNS column order.
    """
    avg = {
        "runs_scored_r15": 4.5, "runs_allowed_r15": 4.5,
        "run_diff_r15": 0.0, "k_rate_r15": 0.20,
        "bb_rate_r15": 0.09, "run_diff_r10": 0.0,
    }
    home_stats = trailing_stats.get(home_team_br, avg)
    away_stats = trailing_stats.get(away_team_br, avg)

    home_rest = max(1, min(int(home_days_rest), 7))
    away_rest = max(1, min(int(away_days_rest), 7))

    pf = PARK_FACTORS.get(home_team_br, {"runs": 100, "hr": 100})

    feat = {
        "home_runs_scored_r15": home_stats["runs_scored_r15"],
        "home_runs_allowed_r15": home_stats["runs_allowed_r15"],
        "home_run_diff_r15": home_stats["run_diff_r15"],
        "home_k_rate_r15": home_stats["k_rate_r15"],
        "home_bb_rate_r15": home_stats["bb_rate_r15"],
        "home_run_diff_r10": home_stats["run_diff_r10"],
        "away_runs_scored_r15": away_stats["runs_scored_r15"],
        "away_runs_allowed_r15": away_stats["runs_allowed_r15"],
        "away_run_diff_r15": away_stats["run_diff_r15"],
        "away_k_rate_r15": away_stats["k_rate_r15"],
        "away_bb_rate_r15": away_stats["bb_rate_r15"],
        "away_run_diff_r10": away_stats["run_diff_r10"],
        "offense_diff": home_stats["runs_scored_r15"] - away_stats["runs_scored_r15"],
        "defense_diff": home_stats["runs_allowed_r15"] - away_stats["runs_allowed_r15"],
        "form_diff": home_stats["run_diff_r10"] - away_stats["run_diff_r10"],
        "is_home": 1,
        "home_days_rest": float(home_rest),
        "away_days_rest": float(away_rest),
        "rest_advantage": float(home_rest - away_rest),
        "park_factor_runs": pf["runs"] / 100.0,
        "park_factor_hr": pf["hr"] / 100.0,
    }
    return pd.DataFrame([feat], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------


def prob_to_american_odds(prob: float) -> int:
    """Convert win probability to American odds integer."""
    prob = max(0.001, min(0.999, prob))
    if prob > 0.5:
        return round(-(prob / (1 - prob)) * 100)
    return round(((1 - prob) / prob) * 100)


def _american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (raw, with vig)."""
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


# ---------------------------------------------------------------------------
# Odds fetching
# ---------------------------------------------------------------------------


def _fetch_mlb_odds() -> list[dict]:
    """Fetch today's MLB odds from The Odds API.

    Returns a list of game dicts, each with:
    home_team, away_team, commence_time, game_id,
    market_home_ml, market_away_ml, market_spread, market_total.
    """
    import os

    from line_tracker.scraper import OddsClient
    from line_tracker.models import BetType

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        logger.warning("ODDS_API_KEY not set; cannot fetch live odds.")
        return []

    try:
        with OddsClient(api_key) as client:
            lines = client.get_odds(sport="baseball_mlb")
    except Exception as exc:
        logger.warning("Failed to fetch MLB odds: %s", exc)
        return []

    # Group by game
    games: dict[str, dict] = {}
    for ln in lines:
        key = f"{ln.home_team}|{ln.away_team}"
        if key not in games:
            ct = ln.commence_time.isoformat() if ln.commence_time else ""
            games[key] = {
                "game_id": ln.api_event_id or key,
                "home_team": ln.home_team,
                "away_team": ln.away_team,
                "commence_time": ct,
                "market_home_ml": None,
                "market_away_ml": None,
                "market_spread": None,
                "market_total": None,
                "_home_ml_list": [],
                "_away_ml_list": [],
                "_spread_list": [],
                "_total_list": [],
            }
        g = games[key]
        if ln.bet_type == BetType.MONEYLINE:
            if ln.home_value is not None:
                g["_home_ml_list"].append(float(ln.home_value))
            if ln.away_value is not None:
                g["_away_ml_list"].append(float(ln.away_value))
        elif ln.bet_type == BetType.SPREAD:
            if ln.home_value is not None:
                g["_spread_list"].append(float(ln.home_value))
        elif ln.bet_type == BetType.TOTAL:
            if ln.home_value is not None:
                g["_total_list"].append(float(ln.home_value))

    result = []
    for g in games.values():
        if g["_home_ml_list"]:
            # Use median as consensus ML
            g["market_home_ml"] = int(round(float(np.median(g["_home_ml_list"]))))
        if g["_away_ml_list"]:
            g["market_away_ml"] = int(round(float(np.median(g["_away_ml_list"]))))
        if g["_spread_list"]:
            g["market_spread"] = float(np.median(g["_spread_list"]))
        if g["_total_list"]:
            g["market_total"] = float(np.median(g["_total_list"]))
        # Clean up internal lists
        for k in ("_home_ml_list", "_away_ml_list", "_spread_list", "_total_list"):
            del g[k]
        result.append(g)

    return result


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------


def predict_mlb_games(
    target_date: date | None = None,
    models_dir: Path | None = None,
) -> list[dict]:
    """Generate MLB model predictions for a given date.

    Parameters
    ----------
    target_date:
        Date to predict. Defaults to today.
    models_dir:
        Directory containing model artifacts.

    Returns
    -------
    List of prediction dicts, one per game.
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime("%Y-%m-%d")

    # Load artifacts
    artifacts = load_mlb_artifacts(models_dir)

    # Fetch odds / games
    odds_games = _fetch_mlb_odds()
    if not odds_games:
        logger.info("No MLB games found in odds feed for %s", date_str)
        return []

    # Filter to target date by commence_time
    target_games = []
    for g in odds_games:
        ct = g.get("commence_time", "")
        if ct and date_str in ct:
            target_games.append(g)
        elif not ct:
            # If no commence_time, include (assume today)
            target_games.append(g)

    if not target_games:
        logger.info("No MLB games for %s after date filter", date_str)
        return []

    # Load trailing stats
    trailing_stats = get_trailing_stats(season=2025)

    predictions: list[dict] = []

    for g in target_games:
        home_full = g["home_team"]
        away_full = g["away_team"]

        home_br = ODDS_TO_BR.get(home_full)
        away_br = ODDS_TO_BR.get(away_full)

        if home_br is None:
            logger.warning("Unknown home team: %r — skipping", home_full)
            continue
        if away_br is None:
            logger.warning("Unknown away team: %r — skipping", away_full)
            continue

        # Build features
        X = build_prediction_features(home_br, away_br, trailing_stats)

        # --- Moneyline ---
        ml_art = artifacts["moneyline"]
        X_ml = ml_art["scaler"].transform(X)
        home_win_prob = float(ml_art["model"].predict_proba(X_ml)[0][1])

        # --- Margin ---
        mg_art = artifacts["margin"]
        X_mg = mg_art["scaler"].transform(X)
        run_diff = float(mg_art["model"].predict(X_mg)[0])

        # --- Totals ---
        tot_art = artifacts["totals"]
        X_tot = tot_art["scaler"].transform(X)
        total_runs = float(tot_art["model"].predict(X_tot)[0])

        # Derived odds
        implied_home_odds = prob_to_american_odds(home_win_prob)

        # Market data
        market_home_ml = g.get("market_home_ml")
        market_away_ml = g.get("market_away_ml")
        market_spread = g.get("market_spread")
        market_total = g.get("market_total")

        # Edges
        if market_home_ml is not None and market_away_ml is not None:
            implied_home = _american_to_prob(market_home_ml)
            implied_away = _american_to_prob(market_away_ml)
            total_implied = implied_home + implied_away
            # De-vig
            devig_home = implied_home / total_implied if total_implied > 0 else 0.5
            ml_edge = home_win_prob - devig_home
        else:
            ml_edge = 0.0

        total_edge = (total_runs - market_total) if market_total is not None else 0.0

        # Spread pick
        if run_diff > 0:
            spread_pick = f"Home -{abs(run_diff):.1f}"
        else:
            spread_pick = f"Away +{abs(run_diff):.1f}"

        # Confidence
        abs_ml = abs(ml_edge)
        abs_total = abs(total_edge)
        if abs_ml >= 0.05 or abs_total >= 1.0:
            confidence = "High"
        elif abs_ml >= 0.03 or abs_total >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        predictions.append({
            "game_id": g["game_id"],
            "game_date": date_str,
            "home_team": home_full,
            "away_team": away_full,
            "home_team_br": home_br,
            "away_team_br": away_br,
            "commence_time": g.get("commence_time", ""),
            "model_home_win_prob": round(home_win_prob, 4),
            "model_run_diff": round(run_diff, 2),
            "model_total_runs": round(total_runs, 2),
            "model_implied_home_odds": implied_home_odds,
            "market_home_ml": market_home_ml,
            "market_away_ml": market_away_ml,
            "market_spread": market_spread,
            "market_total": market_total,
            "ml_edge": round(ml_edge, 4),
            "total_edge": round(total_edge, 2),
            "model_spread_pick": spread_pick,
            "confidence": confidence,
            "data_source": "2025_trailing",
        })

    logger.info("Generated %d MLB predictions for %s", len(predictions), date_str)
    return predictions

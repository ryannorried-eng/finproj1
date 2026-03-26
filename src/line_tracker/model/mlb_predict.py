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

from line_tracker.model.mlb_features import (
    FEATURE_COLUMNS, PARK_FACTORS, _ROT_DEFAULTS, _BP_DEFAULT_ERA,
    LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP, LEAGUE_AVG_K9,
)

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

        # v2 — home/away splits from last N home/away games
        home_games = group[group["home_away"] == "H"].tail(15)
        away_games = group[group["home_away"] == "A"].tail(15)

        rs_home_r15 = float(home_games["runs_scored"].mean()) if home_games["runs_scored"].notna().any() else rs15
        ra_home_r15 = float(home_games["runs_allowed"].mean()) if home_games["runs_allowed"].notna().any() else ra15
        rs_away_r15 = float(away_games["runs_scored"].mean()) if away_games["runs_scored"].notna().any() else rs15
        ra_away_r15 = float(away_games["runs_allowed"].mean()) if away_games["runs_allowed"].notna().any() else ra15

        result[str(team)] = {
            "runs_scored_r15": rs15,
            "runs_allowed_r15": ra15,
            "run_diff_r15": rd15,
            "k_rate_r15": k15,
            "bb_rate_r15": bb15,
            "run_diff_r10": rd10,
            # v2 splits
            "runs_scored_home_r15": rs_home_r15,
            "runs_allowed_home_r15": ra_home_r15,
            "runs_scored_away_r15": rs_away_r15,
            "runs_allowed_away_r15": ra_away_r15,
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
    rotation_lookup: dict | None = None,
    season: int = 2025,
    home_pitcher_stats: dict | None = None,
    away_pitcher_stats: dict | None = None,
    home_bullpen_era: float | None = None,
    away_bullpen_era: float | None = None,
    weather: dict | None = None,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for a game prediction.

    Parameters
    ----------
    home_team_br, away_team_br:
        Team BR abbreviations (e.g. "LAD", "SFG").
    trailing_stats:
        Dict from get_trailing_stats().
    home_days_rest, away_days_rest:
        Days rest, capped at 7.
    rotation_lookup:
        Optional {(team, season): rotation_stats} from _compute_rotation_quality.
    season:
        Season year for rotation lookup.
    home_pitcher_stats, away_pitcher_stats:
        Optional individual SP stats dict with keys: era, k9, whip.
        When provided, overrides rotation averages.
    home_bullpen_era, away_bullpen_era:
        Optional bullpen ERA_r7 values. Falls back to league average.
    weather:
        Optional weather dict with keys: wind_out_factor, temp_f, precip_prob.

    Returns
    -------
    Single-row DataFrame with FEATURE_COLUMNS column order.
    """
    avg = {
        "runs_scored_r15": 4.5, "runs_allowed_r15": 4.5,
        "run_diff_r15": 0.0, "k_rate_r15": 0.20,
        "bb_rate_r15": 0.09, "run_diff_r10": 0.0,
        "runs_scored_home_r15": 4.5, "runs_allowed_home_r15": 4.5,
        "runs_scored_away_r15": 4.5, "runs_allowed_away_r15": 4.5,
    }
    home_stats = trailing_stats.get(home_team_br, avg)
    away_stats = trailing_stats.get(away_team_br, avg)

    home_rest = max(1, min(int(home_days_rest), 7))
    away_rest = max(1, min(int(away_days_rest), 7))

    pf = PARK_FACTORS.get(home_team_br, {"runs": 100, "hr": 100})

    # v2 — rotation quality
    if rotation_lookup is not None:
        home_rot = rotation_lookup.get((home_team_br, season), _ROT_DEFAULTS)
        away_rot = rotation_lookup.get((away_team_br, season), _ROT_DEFAULTS)
    else:
        home_rot = _ROT_DEFAULTS.copy()
        away_rot = _ROT_DEFAULTS.copy()

    # Override rotation with actual probable pitcher if available
    if home_pitcher_stats is not None:
        home_rotation_era = float(home_pitcher_stats.get("era") or home_rot["rotation_era"])
        home_rotation_k9 = float(home_pitcher_stats.get("k9") or home_rot["rotation_k9"])
        home_rotation_whip = float(home_pitcher_stats.get("whip") or home_rot["rotation_whip"])
    else:
        home_rotation_era = home_rot["rotation_era"]
        home_rotation_k9 = home_rot["rotation_k9"]
        home_rotation_whip = home_rot["rotation_whip"]

    if away_pitcher_stats is not None:
        away_rotation_era = float(away_pitcher_stats.get("era") or away_rot["rotation_era"])
        away_rotation_k9 = float(away_pitcher_stats.get("k9") or away_rot["rotation_k9"])
        away_rotation_whip = float(away_pitcher_stats.get("whip") or away_rot["rotation_whip"])
    else:
        away_rotation_era = away_rot["rotation_era"]
        away_rotation_k9 = away_rot["rotation_k9"]
        away_rotation_whip = away_rot["rotation_whip"]

    sp_era_diff = away_rotation_era - home_rotation_era

    # v2 — bullpen
    h_bp_era = float(home_bullpen_era) if home_bullpen_era is not None else _BP_DEFAULT_ERA
    a_bp_era = float(away_bullpen_era) if away_bullpen_era is not None else _BP_DEFAULT_ERA
    bullpen_era_diff = a_bp_era - h_bp_era

    # v2 — home/away splits
    home_rs_home = home_stats.get("runs_scored_home_r15", home_stats["runs_scored_r15"])
    home_ra_home = home_stats.get("runs_allowed_home_r15", home_stats["runs_allowed_r15"])
    away_rs_away = away_stats.get("runs_scored_away_r15", away_stats["runs_scored_r15"])
    away_ra_away = away_stats.get("runs_allowed_away_r15", away_stats["runs_allowed_r15"])

    # v2 — weather (use real values if provided, else neutral)
    if weather is not None:
        wind_out_factor = float(weather.get("wind_out_factor", 0.0))
        temp_f = float(weather.get("temp_f", 72.0))
        precip_prob = float(weather.get("precip_prob", 0.0))
    else:
        wind_out_factor = 0.0
        temp_f = 72.0
        precip_prob = 0.0

    # v3 — per-game SP stats: use pitcher_stats if available, else rotation avg,
    # else league average
    if home_pitcher_stats is not None:
        home_sp_era  = float(home_pitcher_stats.get("era")  or home_rotation_era or LEAGUE_AVG_ERA)
        home_sp_whip = float(home_pitcher_stats.get("whip") or home_rotation_whip or LEAGUE_AVG_WHIP)
        home_sp_k9   = float(home_pitcher_stats.get("k9")   or home_rotation_k9  or LEAGUE_AVG_K9)
    else:
        home_sp_era  = home_rotation_era  if home_rotation_era  else LEAGUE_AVG_ERA
        home_sp_whip = home_rotation_whip if home_rotation_whip else LEAGUE_AVG_WHIP
        home_sp_k9   = home_rotation_k9   if home_rotation_k9   else LEAGUE_AVG_K9

    if away_pitcher_stats is not None:
        away_sp_era  = float(away_pitcher_stats.get("era")  or away_rotation_era or LEAGUE_AVG_ERA)
        away_sp_whip = float(away_pitcher_stats.get("whip") or away_rotation_whip or LEAGUE_AVG_WHIP)
        away_sp_k9   = float(away_pitcher_stats.get("k9")   or away_rotation_k9  or LEAGUE_AVG_K9)
    else:
        away_sp_era  = away_rotation_era  if away_rotation_era  else LEAGUE_AVG_ERA
        away_sp_whip = away_rotation_whip if away_rotation_whip else LEAGUE_AVG_WHIP
        away_sp_k9   = away_rotation_k9   if away_rotation_k9   else LEAGUE_AVG_K9

    sp_era_diff_v3 = away_sp_era - home_sp_era

    from line_tracker.model.mlb_features import LEAGUE_AVG_OPS, LEAGUE_AVG_WRC_PLUS

    feat = {
        # v1
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
        # v3 — per-game starting pitcher stats
        "home_sp_era": home_sp_era,
        "away_sp_era": away_sp_era,
        "home_sp_whip": home_sp_whip,
        "away_sp_whip": away_sp_whip,
        "home_sp_k9": home_sp_k9,
        "away_sp_k9": away_sp_k9,
        "sp_era_diff": sp_era_diff_v3,
        # v2 — bullpen
        "home_bullpen_era_r7": h_bp_era,
        "away_bullpen_era_r7": a_bp_era,
        "bullpen_era_diff": bullpen_era_diff,
        # v2 — splits
        "home_team_runs_scored_home_r15": home_rs_home,
        "away_team_runs_scored_away_r15": away_rs_away,
        "home_team_runs_allowed_home_r15": home_ra_home,
        "away_team_runs_allowed_away_r15": away_ra_away,
        # v2 — weather
        "wind_out_factor": wind_out_factor,
        "temp_f": temp_f,
        "precip_prob": precip_prob,
        # v4 — lineup strength (neutral defaults for prediction; real lineups not yet integrated)
        "home_lineup_wrc":       LEAGUE_AVG_WRC_PLUS,
        "away_lineup_wrc":       LEAGUE_AVG_WRC_PLUS,
        "home_lineup_top3_ops":  LEAGUE_AVG_OPS,
        "away_lineup_top3_ops":  LEAGUE_AVG_OPS,
        "lineup_wrc_diff":       0.0,
        "home_lineup_depth_ops": LEAGUE_AVG_OPS,
        # v4 — SP workload (neutral defaults)
        "home_sp_rest_days": 5.0,
        "away_sp_rest_days": 5.0,
        "home_sp_avg_ip":    5.5,
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

    # Filter to target date by commence_time.
    # The Odds API returns commence_time in UTC. MLB games in the US start in
    # the afternoon/evening local time (ET), so a game at 8 pm ET = 00:05 UTC
    # the *next* day. A raw UTC date-string comparison would incorrectly drop
    # those evening games. Convert to US/Eastern before comparing the date.
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        _ET = ZoneInfo("America/New_York")
    except Exception:
        _ET = None  # fallback handled below

    def _game_local_date(ct: str) -> date | None:
        """Return the US/Eastern calendar date for a UTC ISO commence_time string."""
        if not ct:
            return None
        try:
            dt = datetime.fromisoformat(ct)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if _ET is not None:
                return dt.astimezone(_ET).date()
            # Fallback: use UTC-5 (covers EST; EDT games are UTC-4 so off by at
            # most one hour, which still gives the correct local date for all
            # realistic MLB start times)
            from datetime import timedelta
            return (dt + timedelta(hours=-5)).date()
        except (ValueError, TypeError):
            return None

    target_games = []
    for g in odds_games:
        ct = g.get("commence_time", "")
        local_date = _game_local_date(ct)
        if local_date is not None:
            if local_date == target_date:
                target_games.append(g)
        else:
            # No commence_time — assume today
            target_games.append(g)

    if not target_games:
        logger.info("No MLB games for %s after date filter", date_str)
        return []

    # Load trailing stats and v2 data
    trailing_stats = get_trailing_stats(season=2025)

    # v2 — fetch probable pitchers and pitcher season stats
    from line_tracker.model.mlb_data import (
        fetch_probable_pitchers,
        fetch_pitcher_season_stats,
        fetch_bullpen_stats,
        fetch_weather,
    )
    from line_tracker.model.mlb_features import _compute_rotation_quality

    probables_df: pd.DataFrame | None = None
    try:
        probables_df = fetch_probable_pitchers(target_date)
    except Exception as exc:
        logger.warning("Could not fetch probable pitchers: %s", exc)


    pitcher_stats_df: pd.DataFrame | None = None
    try:
        pitcher_stats_df = fetch_pitcher_season_stats(2025)
    except Exception as exc:
        logger.warning("Could not fetch pitcher stats: %s", exc)

    rotation_lookup: dict = {}
    try:
        rotation_lookup = _compute_rotation_quality([2025])
    except Exception as exc:
        logger.warning("Could not compute rotation quality: %s", exc)

    bullpen_lookup: dict = {}  # team → era_r7
    try:
        bp_df = fetch_bullpen_stats(2025)
        # Use the most recent era_r7 per team
        latest_bp = (
            bp_df.sort_values("date")
            .groupby("team")
            .last()
            .reset_index()
        )
        for row in latest_bp.itertuples(index=False):
            era = float(row.era_r7) if pd.notna(row.era_r7) else 4.20
            bullpen_lookup[str(row.team)] = era
    except Exception as exc:
        logger.warning("Could not fetch bullpen stats: %s", exc)

    # Build probables index: {(home_team, away_team): row}
    probables_index: dict = {}
    if probables_df is not None and not probables_df.empty:
        for row in probables_df.itertuples(index=False):
            key = (str(row.home_team), str(row.away_team))
            probables_index[key] = row

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

        # v2 — look up probable pitchers for this game
        probable_row = probables_index.get((home_br, away_br))
        home_pitcher_stats: dict | None = None
        away_pitcher_stats: dict | None = None
        home_pitcher_name: str | None = None
        away_pitcher_name: str | None = None
        home_pitcher_era: float | None = None
        away_pitcher_era: float | None = None

        if probable_row is not None and pitcher_stats_df is not None:
            # Convert pitcher IDs from float64 (parquet storage) to int,
            # guarding against NaN (no pitcher announced yet).
            h_pid_raw = probable_row.home_pitcher_id
            a_pid_raw = probable_row.away_pitcher_id
            h_pid = int(h_pid_raw) if h_pid_raw is not None and pd.notna(h_pid_raw) else None
            a_pid = int(a_pid_raw) if a_pid_raw is not None and pd.notna(a_pid_raw) else None
            home_pitcher_name = probable_row.home_pitcher_name or None
            away_pitcher_name = probable_row.away_pitcher_name or None

            if h_pid is not None and h_pid in pitcher_stats_df.index:
                row_p = pitcher_stats_df.loc[h_pid]
                home_pitcher_stats = {
                    "era": row_p.get("era") if pd.notna(row_p.get("era")) else None,
                    "k9": row_p.get("k9") if pd.notna(row_p.get("k9")) else None,
                    "whip": row_p.get("whip") if pd.notna(row_p.get("whip")) else None,
                }
                home_pitcher_era = home_pitcher_stats["era"]

            if a_pid is not None and a_pid in pitcher_stats_df.index:
                row_p = pitcher_stats_df.loc[a_pid]
                away_pitcher_stats = {
                    "era": row_p.get("era") if pd.notna(row_p.get("era")) else None,
                    "k9": row_p.get("k9") if pd.notna(row_p.get("k9")) else None,
                    "whip": row_p.get("whip") if pd.notna(row_p.get("whip")) else None,
                }
                away_pitcher_era = away_pitcher_stats["era"]

        # v2 — fetch weather for home park
        weather: dict | None = None
        try:
            weather = fetch_weather(home_br, target_date)
        except Exception as exc:
            logger.debug("Weather fetch failed for %s: %s", home_br, exc)

        # Build features
        X = build_prediction_features(
            home_br, away_br, trailing_stats,
            rotation_lookup=rotation_lookup,
            season=2025,
            home_pitcher_stats=home_pitcher_stats,
            away_pitcher_stats=away_pitcher_stats,
            home_bullpen_era=bullpen_lookup.get(home_br),
            away_bullpen_era=bullpen_lookup.get(away_br),
            weather=weather,
        )

        # --- Moneyline ---
        ml_art = artifacts["moneyline"]
        ml_scaler = ml_art["scaler"]
        if ml_scaler is not None:
            # Convert to numpy array BEFORE scaling to avoid feature name warning
            X_np = X.astype(float).values  # numpy array, no feature names
            X_ml = np.nan_to_num(ml_scaler.transform(X_np), nan=0.0)
        else:
            X_ml = np.nan_to_num(X.astype(float).values, nan=0.0)
        home_win_prob = float(ml_art["model"].predict_proba(X_ml)[0][1])

        # --- Margin ---
        mg_art = artifacts["margin"]
        mg_scaler = mg_art["scaler"]
        if mg_scaler is not None:
            X_mg = np.nan_to_num(mg_scaler.transform(X), nan=0.0)
            run_diff = float(mg_art["model"].predict(X_mg)[0])
        else:
            # Pass DataFrame WITH feature names so HGB can match column order
            X_df = X.astype(float).fillna(0.0)
            run_diff = float(mg_art["model"].predict(X_df)[0])

        # --- Totals ---
        tot_art = artifacts["totals"]
        tot_scaler = tot_art["scaler"]
        if tot_scaler is not None:
            X_tot = np.nan_to_num(tot_scaler.transform(X), nan=0.0)
            total_runs = float(tot_art["model"].predict(X_tot)[0])
        else:
            # Pass DataFrame WITH feature names so HGB can match column order
            X_df_tot = X.astype(float).fillna(0.0)
            total_runs = float(tot_art["model"].predict(X_df_tot)[0])

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
            # v2 fields
            "home_pitcher": home_pitcher_name,
            "away_pitcher": away_pitcher_name,
            "home_pitcher_era": round(home_pitcher_era, 2) if home_pitcher_era is not None else None,
            "away_pitcher_era": round(away_pitcher_era, 2) if away_pitcher_era is not None else None,
            "temp_f": round(weather["temp_f"], 1) if weather else None,
            "wind_mph": round(weather["wind_mph"], 1) if weather else None,
            "wind_out_factor": round(weather["wind_out_factor"], 2) if weather else None,
        })

    logger.info("Generated %d MLB predictions for %s", len(predictions), date_str)
    return predictions

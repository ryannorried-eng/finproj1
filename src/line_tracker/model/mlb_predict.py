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

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message="X has feature names",
    category=UserWarning,
    module="sklearn",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

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

# Teams where 2025 trailing stats are least predictive of 2026 performance
# Based on 4-day sample (46 games) — revisit at 100 games
# Format: frozenset of BR abbreviations for a matchup
BLIND_SPOT_MATCHUPS = [
    frozenset({"WSN", "CHC"}),  # Model 1-2 on Cubs picks, overrating CHC
    frozenset({"KCR", "ATL"}),  # Model 0-2 on Royals picks, underrating ATL
    frozenset({"COL", "MIA"}),  # COL consistently underperforming model
]


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
                if "artifact_file" not in data.get("metadata", {}):
                    data.setdefault("metadata", {})["artifact_file"] = candidates[-1].name
                # sklearn <1.5 predict_proba checks self.multi_class; models
                # trained on sklearn >=1.5 no longer store this attribute.
                # Patch the loaded model so it works across sklearn versions.
                _model = data.get("model")
                if _model is not None and not hasattr(_model, "multi_class"):
                    _model.multi_class = "auto"
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


def fetch_current_season_logs(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch 2026 team game logs from MLB Stats API.
    Uses the same pattern as mlb_data.fetch_team_game_logs()
    but for the current season.

    Cache: ~/.cache/line_tracker/mlb/team_logs_2026.parquet
    Cache expiry: 6 hours (refresh if older than 6 hours)

    Returns DataFrame with same schema as fetch_team_game_logs():
      team, date, home_away, opponent,
      runs_scored, runs_allowed, hits, walks,
      strikeouts, innings_pitched

    Returns empty DataFrame if API call fails.
    """
    from line_tracker.model.mlb_data import (
        CACHE_DIR, MLB_STATS_API, _to_canonical, _get_mlb_teams, _safe_float
    )
    import requests
    import time

    cache_path = CACHE_DIR / "team_logs_2026.parquet"

    # Check cache age — refresh if older than 6 hours
    if not force_refresh and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 6:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

    logger.info("Fetching 2026 team game logs...")

    try:
        teams = _get_mlb_teams()
    except Exception as exc:
        logger.warning("Could not fetch team list: %s", exc)
        return pd.DataFrame()

    all_rows = []

    for team in teams:
        team_id = team.get("id")
        abbrev = team.get("abbreviation", "")
        br_abbrev = _to_canonical(abbrev)

        if not team_id:
            continue

        hitting_splits: dict = {}
        pitching_splits: dict = {}

        for group in ["hitting", "pitching"]:
            try:
                url = (
                    f"{MLB_STATS_API}/teams/{team_id}/stats"
                    f"?stats=gameLog&season=2026&group={group}&gameType=R"
                )
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                if group == "hitting":
                    for sg in data.get("stats", []):
                        for split in sg.get("splits", []):
                            date_str = split.get("date", "")
                            game_num = split.get("gameNumber", 1)
                            stat = split.get("stat", {})
                            opponent_abbrev = split.get("opponent", {}).get("abbreviation", "")
                            is_home = split.get("isHome")
                            hitting_splits[(date_str, game_num)] = {
                                "runs_scored": _safe_float(stat.get("runs")),
                                "hits": _safe_float(stat.get("hits")),
                                "walks": _safe_float(stat.get("baseOnBalls")),
                                "strikeouts": _safe_float(stat.get("strikeOuts")),
                                "opponent": _to_canonical(opponent_abbrev),
                                "is_home": is_home,
                            }

                elif group == "pitching":
                    for sg in data.get("stats", []):
                        for split in sg.get("splits", []):
                            date_str = split.get("date", "")
                            game_num = split.get("gameNumber", 1)
                            stat = split.get("stat", {})
                            pitching_splits[(date_str, game_num)] = {
                                "runs_allowed": _safe_float(stat.get("runs")),
                                "innings_pitched": _safe_float(stat.get("inningsPitched")),
                            }

                time.sleep(0.05)

            except Exception as exc:
                logger.debug("Game log failed for %s %s: %s", br_abbrev, group, exc)
                continue

        # Join hitting + pitching by date/game_num
        for (date_str, game_num), hit in hitting_splits.items():
            pit = pitching_splits.get((date_str, game_num), {})
            home_away = "H" if hit.get("is_home") else "A"
            all_rows.append({
                "team": br_abbrev,
                "date": pd.Timestamp(date_str),
                "home_away": home_away,
                "opponent": hit.get("opponent"),
                "runs_scored": hit.get("runs_scored"),
                "runs_allowed": pit.get("runs_allowed"),
                "hits": hit.get("hits"),
                "walks": hit.get("walks"),
                "strikeouts": hit.get("strikeouts"),
                "innings_pitched": pit.get("innings_pitched"),
            })

    if not all_rows:
        logger.warning("No 2026 game logs retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # Cache result
    df.to_parquet(cache_path, index=False)
    logger.info("Fetched %d 2026 game log rows for %d teams",
                len(df), df["team"].nunique())
    return df


def get_trailing_stats(base_season: int = 2025) -> dict[str, dict]:
    """
    Compute per-team rolling stats for use as prediction features.

    Logic:
    - Fetch 2026 game logs (cached, refreshed every 6 hours)
    - For teams with >= 15 games in 2026: use 2026 rolling window
    - For teams with 5-14 games in 2026: blend 2026 recent + 2025 trailing
    - For teams with < 5 games in 2026: use 2025 end-of-season trailing

    Returns dict: {team_abbrev: {feature: value}}
    """
    from line_tracker.model.mlb_data import (
        CACHE_DIR, fetch_team_game_logs
    )
    from line_tracker.model.mlb_features import compute_team_rolling_stats

    # Step 1: Try to get 2026 game logs
    logs_2026 = pd.DataFrame()
    try:
        logs_2026 = fetch_current_season_logs()
    except Exception as exc:
        logger.warning("Could not fetch 2026 logs: %s", exc)

    # Step 2: Get 2025 end-of-season trailing stats as fallback
    trailing_cache = CACHE_DIR / f"trailing_stats_{base_season}.parquet"

    stats_2025: dict[str, dict] = {}
    try:
        if trailing_cache.exists():
            df_2025 = pd.read_parquet(trailing_cache)
            for _, row in df_2025.iterrows():
                stats_2025[row["team"]] = row.to_dict()
        else:
            logs_base = fetch_team_game_logs(base_season)
            if not logs_base.empty:
                rolling = compute_team_rolling_stats(logs_base)
                for team, group in rolling.groupby("team"):
                    last = group.sort_values("date").iloc[-1]
                    stats_2025[str(team)] = {
                        "runs_scored_r15": last.get("runs_scored_r15", 4.5),
                        "runs_allowed_r15": last.get("runs_allowed_r15", 4.5),
                        "run_diff_r15":     last.get("run_diff_r15", 0.0),
                        "k_rate_r15":       last.get("k_rate_r15", 0.21),
                        "bb_rate_r15":      last.get("bb_rate_r15", 0.085),
                        "run_diff_r10":     last.get("run_diff_r10", 0.0),
                    }
    except Exception as exc:
        logger.warning("Could not load %d trailing stats: %s", base_season, exc)

    # Step 3: Build final trailing stats per team
    result: dict[str, dict] = {}

    # Get all known teams
    all_teams: set[str] = set(stats_2025.keys())
    if not logs_2026.empty:
        all_teams |= set(logs_2026["team"].unique())

    for team in all_teams:
        # Count 2026 games for this team
        team_2026 = pd.DataFrame()
        if not logs_2026.empty and "team" in logs_2026.columns:
            team_2026 = logs_2026[logs_2026["team"] == team].sort_values("date")

        n_games_2026 = len(team_2026)

        # Blend threshold: require 8+ games before mixing 2026 data
        # Rationale: 4-5 game samples are too small and prone to outlier
        # skew (e.g. one blowout loss inflates/deflates rolling stats).
        # Cap blend at 50% until full 15-game window to prevent
        # early-season noise from dominating predictions.
        # Revisit threshold at 50+ games if model accuracy degrades.
        if n_games_2026 >= 15:
            # Full 2026 rolling window
            recent = team_2026.tail(15)
            source = "2026_rolling"
        elif n_games_2026 >= 8:
            # Partial blend — capped at 50% 2026 weight
            # 4-5 game samples too small and prone to outlier skew
            source = "2026_blend"
            recent = team_2026
        else:
            # Fewer than 8 games — pure 2025 trailing
            # Not enough 2026 data to trust yet
            recent = pd.DataFrame()
            source = "2025_trailing"

        if not recent.empty:
            rs = pd.to_numeric(recent["runs_scored"], errors="coerce").fillna(0)
            ra = pd.to_numeric(recent["runs_allowed"], errors="coerce").fillna(0)
            h  = pd.to_numeric(recent.get("hits",       pd.Series([0] * len(recent))), errors="coerce").fillna(0)
            bb = pd.to_numeric(recent.get("walks",      pd.Series([0] * len(recent))), errors="coerce").fillna(0)
            so = pd.to_numeric(recent.get("strikeouts", pd.Series([0] * len(recent))), errors="coerce").fillna(0)

            denom = (h + bb + so).replace(0, 1)

            runs_scored_r15  = float(rs.mean())
            runs_allowed_r15 = float(ra.mean())
            run_diff_r15     = float((rs - ra).mean())
            k_rate_r15       = float((so / denom).mean())
            bb_rate_r15      = float((bb / denom).mean())

            last10 = team_2026.tail(10) if n_games_2026 >= 10 else recent
            rs10 = pd.to_numeric(last10["runs_scored"], errors="coerce").fillna(0)
            ra10 = pd.to_numeric(last10["runs_allowed"], errors="coerce").fillna(0)
            run_diff_r10 = float((rs10 - ra10).mean())

            if source == "2026_blend" and team in stats_2025:
                s25 = stats_2025[team]
                # Gradual blend capped at 50% until full 15-game window
                raw_weight = (n_games_2026 - 7) / 8.0  # 8 games = 12.5%, 14 games = 87.5%
                w26 = min(raw_weight, 0.50)            # cap at 50%
                w25 = 1 - w26
                runs_scored_r15  = w26 * runs_scored_r15  + w25 * s25.get("runs_scored_r15", 4.5)
                runs_allowed_r15 = w26 * runs_allowed_r15 + w25 * s25.get("runs_allowed_r15", 4.5)
                run_diff_r15     = w26 * run_diff_r15     + w25 * s25.get("run_diff_r15", 0.0)
                k_rate_r15       = w26 * k_rate_r15       + w25 * s25.get("k_rate_r15", 0.21)
                bb_rate_r15      = w26 * bb_rate_r15      + w25 * s25.get("bb_rate_r15", 0.085)
                run_diff_r10     = w26 * run_diff_r10     + w25 * s25.get("run_diff_r10", 0.0)

            # Preserve home/away splits from 2025 if available (blend doesn't compute them)
            s_base = stats_2025.get(team, {})
            result[team] = {
                "runs_scored_r15":        runs_scored_r15,
                "runs_allowed_r15":       runs_allowed_r15,
                "run_diff_r15":           run_diff_r15,
                "k_rate_r15":             k_rate_r15,
                "bb_rate_r15":            bb_rate_r15,
                "run_diff_r10":           run_diff_r10,
                "runs_scored_home_r15":   s_base.get("runs_scored_home_r15", runs_scored_r15),
                "runs_allowed_home_r15":  s_base.get("runs_allowed_home_r15", runs_allowed_r15),
                "runs_scored_away_r15":   s_base.get("runs_scored_away_r15", runs_scored_r15),
                "runs_allowed_away_r15":  s_base.get("runs_allowed_away_r15", runs_allowed_r15),
                "data_source":            source,
                "games_2026":             n_games_2026,
            }

        elif team in stats_2025:
            s = stats_2025[team].copy()
            s["data_source"] = "2025_trailing"
            s["games_2026"] = 0
            result[team] = s

    logger.info(
        "Trailing stats: %d teams | "
        "2026_rolling: %d | 2026_blend: %d | 2025_trailing: %d",
        len(result),
        sum(1 for v in result.values() if v.get("data_source") == "2026_rolling"),
        sum(1 for v in result.values() if v.get("data_source") == "2026_blend"),
        sum(1 for v in result.values() if v.get("data_source") == "2025_trailing"),
    )

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
                "market_home_spread_odds": None,
                "market_away_spread_odds": None,
                "market_total": None,
                "_home_ml_list": [],
                "_away_ml_list": [],
                "_spread_list": [],
                "_home_spread_price_list": [],
                "_away_spread_price_list": [],
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
            if ln.home_price is not None:
                g["_home_spread_price_list"].append(float(ln.home_price))
            if ln.away_price is not None:
                g["_away_spread_price_list"].append(float(ln.away_price))
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
        if g["_home_spread_price_list"]:
            g["market_home_spread_odds"] = int(round(float(np.median(g["_home_spread_price_list"]))))
        if g["_away_spread_price_list"]:
            g["market_away_spread_odds"] = int(round(float(np.median(g["_away_spread_price_list"]))))
        if g["_total_list"]:
            g["market_total"] = float(np.median(g["_total_list"]))
        # Clean up internal lists
        for k in (
            "_home_ml_list", "_away_ml_list", "_spread_list",
            "_home_spread_price_list", "_away_spread_price_list", "_total_list",
        ):
            del g[k]
        result.append(g)

    return result


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------


def predict_mlb_games(
    target_date: date | None = None,
    models_dir: Path | None = None,
    use_ensemble: bool = True,
) -> list[dict]:
    """Generate MLB model predictions for a given date.

    Parameters
    ----------
    target_date:
        Date to predict. Defaults to today.
    models_dir:
        Directory containing model artifacts.
    use_ensemble:
        If True (default), enrich predictions with ensemble meta-model outputs.
        Falls back gracefully to base moneyline if ensemble artifacts are absent.

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
    trailing_stats = get_trailing_stats()

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

    # Per-run weather deduplication: avoids re-fetching the same ballpark
    # within one predict_mlb_games call (e.g. doubleheaders, repeated calls).
    # The module-level TTL cache in mlb_data.py handles cross-call deduplication.
    _WEATHER_SENTINEL = object()
    _weather_run_cache: dict[str, dict | None] = {}

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

        # v2 — fetch weather for home park (deduplicated within this run)
        weather: dict | None = _weather_run_cache.get(home_br, _WEATHER_SENTINEL)
        if weather is _WEATHER_SENTINEL:
            try:
                weather = fetch_weather(home_br, target_date)
            except Exception as exc:
                logger.warning("Weather fetch failed for %s: %s", home_br, exc)
                weather = None
            _weather_run_cache[home_br] = weather

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

        # Resolve large contradictions: when |run_diff| > 1.0 and the two models
        # disagree directionally, trust the margin model and nudge home_win_prob.
        if abs(run_diff) > 1.0:
            if run_diff > 0 and home_win_prob < 0.5:
                home_win_prob = max(home_win_prob, 0.5 + abs(run_diff) * 0.02)
            elif run_diff < 0 and home_win_prob > 0.5:
                home_win_prob = min(home_win_prob, 0.5 - abs(run_diff) * 0.02)

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
        market_home_spread_odds = g.get("market_home_spread_odds")
        market_away_spread_odds = g.get("market_away_spread_odds")
        market_total = g.get("market_total")

        # Edges
        devig_home: float | None = None
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

        # ML bet qualified: favored team >= 57% AND |ml_edge| >= 0.04
        away_win_prob = 1 - home_win_prob
        favored_prob = max(home_win_prob, away_win_prob)
        ml_bet_qualified = favored_prob >= 0.57 and abs(ml_edge) >= 0.04

        # Blind spot check
        matchup_set = frozenset([home_br, away_br])
        is_blind_spot = any(matchup_set == bs for bs in BLIND_SPOT_MATCHUPS)

        if is_blind_spot:
            # Downgrade confidence by one level
            if confidence == "High":
                confidence = "Medium"
            elif confidence == "Medium":
                confidence = "Low"

        # Determine data_source from trailing stats for each team
        home_source = trailing_stats.get(home_br, {}).get("data_source", "2025_trailing")
        away_source = trailing_stats.get(away_br, {}).get("data_source", "2025_trailing")
        home_games_2026 = trailing_stats.get(home_br, {}).get("games_2026", 0)
        away_games_2026 = trailing_stats.get(away_br, {}).get("games_2026", 0)

        if "rolling" in home_source and "rolling" in away_source:
            data_source = "2026_rolling"
        elif home_games_2026 + away_games_2026 > 0:
            data_source = f"2026_blend ({home_games_2026 + away_games_2026} games)"
        else:
            data_source = "2025_trailing"

        pred = {
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
            "market_home_spread_odds": market_home_spread_odds,
            "market_away_spread_odds": market_away_spread_odds,
            "market_total": market_total,
            "ml_edge": round(ml_edge, 4),
            "total_edge": round(total_edge, 2),
            "model_spread_pick": spread_pick,
            "confidence": confidence,
            "ml_bet_qualified": ml_bet_qualified,
            "blind_spot": is_blind_spot,
            "data_source": data_source,
            # v2 fields
            "home_pitcher": home_pitcher_name,
            "away_pitcher": away_pitcher_name,
            "home_pitcher_era": round(home_pitcher_era, 2) if home_pitcher_era is not None else None,
            "away_pitcher_era": round(away_pitcher_era, 2) if away_pitcher_era is not None else None,
            "home_pitcher_id": h_pid,
            "away_pitcher_id": a_pid,
            "temp_f": round(weather["temp_f"], 1) if weather else None,
            "wind_mph": round(weather["wind_mph"], 1) if weather else None,
            "wind_out_factor": round(weather["wind_out_factor"], 2) if weather else None,
        }

        # ------------------------------------------------------------------
        # Ensemble enrichment (additive — does not break base predictions)
        # ------------------------------------------------------------------
        if use_ensemble:
            try:
                from line_tracker.model.mlb_ensemble import predict_ensemble as _ens
                pred = _ens(pred)

                # If ensemble produced a recommended_prob, recalculate edge
                # using it instead of the raw moneyline probability.
                rec_prob = pred.get("recommended_prob")
                if rec_prob is not None and devig_home is not None:
                    new_ml_edge = rec_prob - devig_home
                    pred["ml_edge"] = round(new_ml_edge, 4)

                    # Recalculate confidence and ml_bet_qualified
                    abs_new_ml = abs(new_ml_edge)
                    if abs_new_ml >= 0.05 or abs(total_edge) >= 1.0:
                        new_conf = "High"
                    elif abs_new_ml >= 0.03 or abs(total_edge) >= 0.6:
                        new_conf = "Medium"
                    else:
                        new_conf = "Low"
                    if is_blind_spot:
                        if new_conf == "High":
                            new_conf = "Medium"
                        elif new_conf == "Medium":
                            new_conf = "Low"
                    pred["confidence"] = new_conf

                    away_win_prob_ens = 1 - rec_prob
                    favored_prob_ens = max(rec_prob, away_win_prob_ens)
                    pred["ml_bet_qualified"] = (
                        favored_prob_ens >= 0.57 and abs_new_ml >= 0.04
                    )

            except Exception as exc:
                logger.debug("Ensemble enrichment failed for %s @ %s: %s",
                             away_full, home_full, exc)

        predictions.append(pred)

    logger.info("Generated %d MLB predictions for %s", len(predictions), date_str)
    return predictions

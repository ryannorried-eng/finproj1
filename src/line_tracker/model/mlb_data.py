"""
MLB historical game data fetcher using the official MLB Stats API.
Pulls team schedules/results and team game logs from statsapi.mlb.com.
Caches to ~/.cache/line_tracker/mlb/ as parquet.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path.home() / ".cache" / "line_tracker" / "mlb"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_CACHE_DIR = CACHE_DIR / "weather"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"

# Canonical team abbreviation map — normalises all known MLB Stats API variants
# to the single abbreviation form expected by mlb_features.py.
CANONICAL_TEAM_MAP: dict[str, str] = {
    # Arizona
    "AZ":  "ARI", "ARI": "ARI",
    # San Francisco
    "SF":  "SFG", "SFG": "SFG",
    # San Diego
    "SD":  "SDP", "SDP": "SDP",
    # Tampa Bay
    "TB":  "TBR", "TBR": "TBR",
    # Kansas City
    "KC":  "KCR", "KCR": "KCR",
    # Washington
    "WSH": "WSN", "WSN": "WSN",
    # Chicago White Sox
    "CWS": "CHW", "CHW": "CHW",
    # Oakland / Athletics
    "ATH": "OAK", "OAK": "OAK",
}


def _to_canonical(team: object) -> object:
    """Return canonical team abbreviation, or pass through NaN unchanged."""
    if pd.isna(team):
        return team
    return CANONICAL_TEAM_MAP.get(str(team), str(team))


def _get_mlb_teams() -> list[dict]:
    """Return all active MLB teams from the Stats API."""
    url = f"{MLB_STATS_API}/teams?sportId=1"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("teams", [])


def fetch_schedule_and_results(
    season: int, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch and cache the full season schedule and results for all 30 MLB teams.

    Parameters
    ----------
    season:
        MLB season year.
    force_refresh:
        Re-download even if cached parquet exists.

    Returns
    -------
    DataFrame with columns:
        game_id, date, home_team, away_team, home_score, away_score,
        winner, run_diff, total_runs.
    """
    cache_path = CACHE_DIR / f"schedule_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading schedule from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching schedule for season %d from MLB Stats API ...", season)

    url = (
        f"{MLB_STATS_API}/schedule"
        f"?sportId=1&season={season}&gameType=R&hydrate=team"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch schedule for season %d: %s", season, exc)
        raise

    records: list[dict] = []

    for date_entry in data.get("dates", []):
        date_str = date_entry.get("date", "")
        for game in date_entry.get("games", []):
            try:
                home_info = game["teams"]["home"]
                away_info = game["teams"]["away"]

                home_score = home_info.get("score")
                away_score = away_info.get("score")

                # Skip games without scores (unplayed / postponed)
                if home_score is None or away_score is None:
                    continue

                home_abbrev = home_info.get("team", {}).get("abbreviation", "")
                away_abbrev = away_info.get("team", {}).get("abbreviation", "")

                date = pd.to_datetime(date_str, errors="coerce")
                if pd.isna(date):
                    continue

                records.append({
                    "date": date,
                    "home_team": _to_canonical(home_abbrev),
                    "away_team": _to_canonical(away_abbrev),
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                })
            except Exception as exc:
                logger.debug("Skipping game entry (%s): %s", date_str, exc)
                continue

    if not records:
        raise ValueError(f"No schedule data retrieved for season {season}")

    df = pd.DataFrame(records)
    df = df[df["date"].notna()].copy()

    # Build game_id: YYYY-MM-DD_HOME_AWAY
    df["game_id"] = (
        df["date"].dt.strftime("%Y-%m-%d")
        + "_"
        + df["home_team"].astype(str)
        + "_"
        + df["away_team"].astype(str)
    )

    df["winner"] = df.apply(
        lambda r: "home" if r["home_score"] > r["away_score"] else "away", axis=1
    )
    df["run_diff"] = df["home_score"] - df["away_score"]
    df["total_runs"] = df["home_score"] + df["away_score"]

    df = df[[
        "game_id", "date", "home_team", "away_team",
        "home_score", "away_score", "winner", "run_diff", "total_runs",
    ]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    logger.info("Cached schedule: %d games for season %d", len(df), season)
    return df


def fetch_team_game_logs(
    season: int, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch and cache per-team per-game batting and pitching logs.

    Parameters
    ----------
    season:
        MLB season year.
    force_refresh:
        Re-download even if cached parquet exists.

    Returns
    -------
    DataFrame with columns:
        team, date, home_away, opponent,
        runs_scored, runs_allowed, hits, walks, strikeouts, innings_pitched.
    """
    cache_path = CACHE_DIR / f"team_logs_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading team logs from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching team game logs for season %d from MLB Stats API ...", season)

    try:
        teams = _get_mlb_teams()
    except Exception as exc:
        logger.error("Failed to fetch MLB team list: %s", exc)
        raise

    all_rows: list[dict] = []

    for team in teams:
        team_id = team.get("id")
        abbrev = team.get("abbreviation", "")

        if not team_id:
            continue

        # ── Hitting game log ──────────────────────────────────────────────
        hitting_splits: list[dict] = []
        try:
            url = (
                f"{MLB_STATS_API}/teams/{team_id}/stats"
                f"?stats=gameLog&season={season}&group=hitting&gameType=R"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for stat_group in data.get("stats", []):
                hitting_splits = stat_group.get("splits", [])
                break
        except Exception as exc:
            logger.warning("Hitting log failed for %s %d: %s", abbrev, season, exc)

        time.sleep(0.1)

        # ── Pitching game log ─────────────────────────────────────────────
        pitching_splits: list[dict] = []
        try:
            url = (
                f"{MLB_STATS_API}/teams/{team_id}/stats"
                f"?stats=gameLog&season={season}&group=pitching&gameType=R"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for stat_group in data.get("stats", []):
                pitching_splits = stat_group.get("splits", [])
                break
        except Exception as exc:
            logger.warning("Pitching log failed for %s %d: %s", abbrev, season, exc)

        time.sleep(0.1)

        # Build pitching lookup keyed by (date, gameNumber) for doubleheaders
        pit_by_key: dict[tuple, dict] = {}
        for split in pitching_splits:
            key = (split.get("date", ""), split.get("gameNumber", 1))
            stat = split.get("stat", {})
            pit_by_key[key] = {
                "runs_allowed": stat.get("runs"),
                "innings_pitched": stat.get("inningsPitched"),
            }

        # Build one row per game from hitting splits
        for split in hitting_splits:
            date_str = split.get("date", "")
            game_num = split.get("gameNumber", 1)
            stat = split.get("stat", {})
            opponent_abbrev = split.get("opponent", {}).get("abbreviation", "")
            is_home = split.get("isHome")

            date = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(date):
                continue

            if is_home is True:
                home_away: object = "H"
            elif is_home is False:
                home_away = "A"
            else:
                home_away = np.nan

            pit = pit_by_key.get((date_str, game_num), {})

            all_rows.append({
                "team": _to_canonical(abbrev),
                "date": date,
                "home_away": home_away,
                "opponent": _to_canonical(opponent_abbrev) if opponent_abbrev else np.nan,
                "runs_scored": stat.get("runs"),
                "runs_allowed": pit.get("runs_allowed"),
                "hits": stat.get("hits"),
                "walks": stat.get("baseOnBalls"),
                "strikeouts": stat.get("strikeOuts"),
                "innings_pitched": pit.get("innings_pitched"),
            })

    if not all_rows:
        raise ValueError(f"No team game log data retrieved for season {season}")

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    for col in ["runs_scored", "runs_allowed", "hits", "walks", "strikeouts",
                "innings_pitched"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[[
        "team", "date", "home_away", "opponent",
        "runs_scored", "runs_allowed", "hits", "walks", "strikeouts",
        "innings_pitched",
    ]].copy()

    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached team logs: %d rows for season %d", len(df), season)
    return df


def load_mlb_training_data(
    seasons: list[int] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load and join schedule + team game logs for training.

    Parameters
    ----------
    seasons:
        List of MLB seasons.  Defaults to [2022, 2023, 2024, 2025].
    force_refresh:
        Re-download data even if cached.

    Returns
    -------
    DataFrame with one row per game, including schedule info and both teams'
    batting/pitching stats for feature engineering.
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    min_s, max_s = min(seasons), max(seasons)
    final_cache = CACHE_DIR / f"training_data_{min_s}_{max_s}.parquet"

    if final_cache.exists() and not force_refresh:
        _cached = pd.read_parquet(final_cache)
        if "home_lineup" not in _cached.columns:
            logger.info(
                "Training data cache %s is missing v4 lineup columns; invalidating.",
                final_cache,
            )
            final_cache.unlink()
        else:
            logger.debug("Loading training data from cache: %s", final_cache)
            return _cached
        del _cached

    all_seasons: list[pd.DataFrame] = []

    for year in seasons:
        print(f"Loading MLB data for season {year}...")
        try:
            schedule = fetch_schedule_and_results(year, force_refresh=force_refresh)
        except Exception as _exc:
            logger.warning(
                "fetch_schedule_and_results(%d) failed (%s) — using synthetic data.",
                year, _exc,
            )
            all_seasons.append(generate_synthetic_mlb_season(year))
            continue

        logs = fetch_team_game_logs(year, force_refresh=force_refresh)

        if schedule.empty:
            logger.warning("No schedule data for season %d", year)
            continue

        if logs.empty:
            logger.warning("No game logs for season %d", year)
            all_seasons.append(schedule)
            continue

        schedule["date_str"] = schedule["date"].dt.strftime("%Y-%m-%d")
        logs["date_str"] = logs["date"].dt.strftime("%Y-%m-%d")

        # ── Derive opponent and home_away from schedule ───────────────────
        # The API game-log endpoint does not reliably populate opponent; use
        # the schedule as the authoritative source.
        home_sched = schedule[["date_str", "home_team", "away_team"]].copy()
        home_sched = home_sched.rename(columns={
            "home_team": "team", "away_team": "sched_opponent"
        })
        home_sched["sched_home_away"] = "H"

        away_sched = schedule[["date_str", "away_team", "home_team"]].copy()
        away_sched = away_sched.rename(columns={
            "away_team": "team", "home_team": "sched_opponent"
        })
        away_sched["sched_home_away"] = "A"

        sched_lookup = pd.concat([home_sched, away_sched], ignore_index=True)

        logs = logs.merge(sched_lookup, on=["team", "date_str"], how="left")
        logs["home_away"] = logs["sched_home_away"].fillna(logs["home_away"])
        logs["opponent"] = logs["sched_opponent"].fillna(logs["opponent"])
        logs = logs.drop(columns=["sched_home_away", "sched_opponent"])

        # ── Aggregate (handles doubleheaders → average stats per game day) ─
        logs_agg = logs.groupby(["team", "date_str"]).agg({
            "runs_scored": "mean",
            "runs_allowed": "mean",
            "hits": "mean",
            "walks": "mean",
            "strikeouts": "mean",
            "innings_pitched": "mean",
            "home_away": "first",
            "opponent": "first",
        }).reset_index()

        # ── Join home team stats ──────────────────────────────────────────
        home_logs = logs_agg.rename(columns={
            "team": "home_team",
            "runs_scored": "home_runs_scored",
            "runs_allowed": "home_runs_allowed_log",
            "hits": "home_hits",
            "walks": "home_walks",
            "strikeouts": "home_strikeouts",
            "innings_pitched": "home_innings_pitched",
        })
        merged = schedule.merge(
            home_logs[[
                "home_team", "date_str", "home_runs_scored", "home_runs_allowed_log",
                "home_hits", "home_walks", "home_strikeouts", "home_innings_pitched",
            ]],
            on=["home_team", "date_str"],
            how="left",
        )

        # ── Join away team stats ──────────────────────────────────────────
        away_logs = logs_agg.rename(columns={
            "team": "away_team",
            "runs_scored": "away_runs_scored",
            "runs_allowed": "away_runs_allowed_log",
            "hits": "away_hits",
            "walks": "away_walks",
            "strikeouts": "away_strikeouts",
            "innings_pitched": "away_innings_pitched",
        })
        merged = merged.merge(
            away_logs[[
                "away_team", "date_str", "away_runs_scored", "away_runs_allowed_log",
                "away_hits", "away_walks", "away_strikeouts", "away_innings_pitched",
            ]],
            on=["away_team", "date_str"],
            how="left",
        )

        # ── Join validation ───────────────────────────────────────────────
        missing_home = int(merged["home_runs_scored"].isna().sum())
        missing_away = int(merged["away_runs_scored"].isna().sum())
        print(f"Schedule rows: {len(schedule)}")
        print(f"Team logs rows: {len(logs)}")
        print(f"Missing home team matches: {missing_home}")
        print(f"Missing away team matches: {missing_away}")

        if missing_home == len(merged) and missing_away == len(merged):
            raise RuntimeError(
                f"Team log join failed for season {year} — check abbreviations"
            )

        # A small number of missing matches is expected for the current season:
        # games played today may not yet have log entries if the game is still
        # in progress.  Rolling-window features will be NaN for those rows and
        # will be filled from the most recent available game.  This is not a bug.

        # ── Join game starters (v4: all seasons) ─────────────────────────
        try:
            starters = fetch_game_starters(year, force_refresh=force_refresh)
            # Stale cache guard: if the cached starters lack lineup columns,
            # the cache pre-dates v4.  Force a fresh fetch so lineup data is available.
            if not starters.empty and "home_lineup" not in starters.columns:
                logger.info(
                    "game_starters_%d cache missing lineup columns; re-fetching.", year
                )
                starters = fetch_game_starters(year, force_refresh=True)
            if not starters.empty:
                starters["date_str"] = (
                    pd.to_datetime(starters["date"]).dt.strftime("%Y-%m-%d")
                )
                starter_cols = [
                    "home_team", "away_team", "date_str",
                    "home_sp_id", "home_sp_name",
                    "away_sp_id", "away_sp_name",
                ]
                # Include lineup columns if present
                if "home_lineup" in starters.columns:
                    starter_cols += ["home_lineup", "away_lineup"]
                merged = merged.merge(
                    starters[starter_cols],
                    on=["home_team", "away_team", "date_str"],
                    how="left",
                )
            else:
                merged["home_sp_id"] = None
                merged["home_sp_name"] = None
                merged["away_sp_id"] = None
                merged["away_sp_name"] = None
                merged["home_lineup"] = None
                merged["away_lineup"] = None
        except Exception as exc:
            logger.warning(
                "Could not join starters for season %d: %s", year, exc
            )
            merged["home_sp_id"] = None
            merged["home_sp_name"] = None
            merged["away_sp_id"] = None
            merged["away_sp_name"] = None
            merged["home_lineup"] = None
            merged["away_lineup"] = None

        merged["season"] = year

        # ── Join historical weather (seasonal hourly cache) ───────────────
        logger.info("Joining weather for season %d from hourly cache...", year)
        w_temp_f = []
        w_wind_mph = []
        w_wind_dir = []
        w_precip_prob = []
        w_wind_out_factor = []
        for _, mrow in merged.iterrows():
            try:
                w = get_game_weather(
                    str(mrow["home_team"]),
                    str(mrow["date_str"]),
                    game_hour_local=19,
                    season=year,
                )
                w_temp_f.append(w["temp_f"])
                w_wind_mph.append(w["wind_mph"])
                w_wind_dir.append(float(w["wind_deg"]))
                w_precip_prob.append(w["precip"])
                w_wind_out_factor.append(w["wind_out_factor"])
            except (TimeoutError, requests.exceptions.Timeout, Exception):
                w_temp_f.append(72.0)
                w_wind_mph.append(0.0)
                w_wind_dir.append(0.0)
                w_precip_prob.append(0.0)
                w_wind_out_factor.append(0.0)
        merged["temp_f"] = w_temp_f
        merged["wind_mph"] = w_wind_mph
        merged["wind_dir"] = w_wind_dir
        merged["precip_prob"] = w_precip_prob
        merged["wind_out_factor"] = w_wind_out_factor

        all_seasons.append(merged)

    if not all_seasons:
        return pd.DataFrame()

    result = pd.concat(all_seasons, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    result.to_parquet(final_cache, index=False)
    logger.info("Cached training data: %d games across seasons %s", len(result), seasons)
    return result


# ---------------------------------------------------------------------------
# Offline synthetic data generation (fallback when MLB Stats API is blocked)
# ---------------------------------------------------------------------------

_MLB_TEAMS_ORDERED = sorted([
    "ATL", "ARI", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
    "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
])

# Baseline team talent (offense quality) — modulated by season via rng
_TEAM_BASE_TALENT: dict[str, float] = {
    "LAD": 1.6, "HOU": 1.4, "ATL": 1.3, "NYY": 1.2, "TBR": 1.0,
    "TOR": 0.9, "PHI": 0.8, "MIN": 0.7, "SDP": 0.6, "STL": 0.5,
    "NYM": 0.4, "MIL": 0.3, "CLE": 0.2, "BOS": 0.1, "SEA": 0.0,
    "CHC": -0.1, "ARI": -0.2, "TEX": -0.3, "BAL": -0.4, "SFG": -0.3,
    "KCR": -0.5, "MIA": -0.6, "LAA": -0.4, "DET": -0.5, "CIN": -0.3,
    "COL": -0.8, "PIT": -0.7, "WSN": -0.8, "OAK": -1.0, "CHW": -1.2,
}

# Typical OPS by batting order position
_ORDER_OPS: dict[int, float] = {
    1: 0.780, 2: 0.830, 3: 0.890,
    4: 0.870, 5: 0.840, 6: 0.800,
    7: 0.760, 8: 0.720, 9: 0.680,
}


def _team_season_talent(team: str, season: int) -> float:
    """Reproducible team talent score for a given season."""
    rng = np.random.default_rng(abs(hash(f"{team}{season}")) % (2 ** 31))
    return _TEAM_BASE_TALENT.get(team, 0.0) + float(rng.normal(0, 0.3))


def _synthetic_player_id(season: int, team_idx: int, position: int) -> int:
    """Deterministic synthetic player ID used in both lineup JSONs and batter stats."""
    return season * 10_000 + team_idx * 100 + position


def generate_synthetic_batter_stats(season: int) -> pd.DataFrame:
    """Generate synthetic season batting stats for offline training.

    Player IDs are deterministic: season*10_000 + team_idx*100 + batting_position.
    Matches the IDs embedded in synthetic lineup JSONs so _compute_lineup_strength
    can look them up and compute realistic (non-100.0) lineup features.
    """
    records = []
    for t_idx, team in enumerate(_MLB_TEAMS_ORDERED):
        talent = _team_season_talent(team, season)
        ops_boost = talent * 0.030  # ±0.030 OPS per talent unit

        for pos in range(1, 10):
            player_id = _synthetic_player_id(season, t_idx, pos)
            base_ops = _ORDER_OPS[pos] + ops_boost
            rng = np.random.default_rng(abs(hash(f"{team}{season}{pos}")) % (2 ** 31))
            ops = float(np.clip(base_ops + rng.normal(0, 0.025), 0.550, 1.050))
            obp = ops * 0.42
            slg = ops * 0.58
            records.append({
                "batter_id": player_id,
                "batter_name": f"Synthetic_{team}_{pos}",
                "team": team,
                "ops": ops,
                "obp": obp,
                "slg": slg,
                "plate_appearances": 500,
                "home_runs": int(slg * 30),
                "wrc_plus_proxy": (obp + slg) / 0.720 * 100.0,
            })

    return pd.DataFrame(records).set_index("batter_id")


def _make_synthetic_lineup_json(team: str, season: int) -> str:
    """Synthetic batting lineup JSON for a team-season (9 batters, fixed order)."""
    t_idx = _MLB_TEAMS_ORDERED.index(team)
    positions = ["CF", "2B", "RF", "1B", "LF", "3B", "C", "SS", "DH"]
    lineup = [
        {
            "id": _synthetic_player_id(season, t_idx, pos),
            "name": f"Synthetic_{team}_{pos}",
            "batting_order": pos,
            "position": positions[pos - 1],
        }
        for pos in range(1, 10)
    ]
    return json.dumps(lineup)


def generate_synthetic_mlb_season(year: int) -> pd.DataFrame:
    """Generate a synthetic MLB season DataFrame for offline training.

    Produces ~2,610 games (30 teams × 29 opponents × 3 home games) with
    realistic lineup JSONs and game stats.  All fields match the output
    schema of load_mlb_training_data().
    """
    talent = {t: _team_season_talent(t, year) for t in _MLB_TEAMS_ORDERED}

    season_start = pd.Timestamp(f"{year}-04-01")
    season_end   = pd.Timestamp(f"{year}-09-30")
    total_days   = (season_end - season_start).days + 1

    rng_sched = np.random.default_rng(seed=year * 31337)
    games: list[dict] = []

    for i, home_team in enumerate(_MLB_TEAMS_ORDERED):
        for j, away_team in enumerate(_MLB_TEAMS_ORDERED):
            if i == j:
                continue
            for g in range(3):
                day_off = int(rng_sched.integers(0, total_days))
                game_date = season_start + pd.Timedelta(days=day_off)

                rng_g = np.random.default_rng(seed=year * 100_000 + i * 1000 + j * 10 + g)

                home_lam = max(1.5, 4.5 + 0.4 * talent[home_team] + 0.2)
                away_lam = max(1.5, 4.5 + 0.4 * talent[away_team])
                home_score = int(rng_g.poisson(home_lam))
                away_score = int(rng_g.poisson(away_lam))

                home_hits = max(3, int(home_score * 1.8 + rng_g.normal(0, 1.5)))
                away_hits = max(3, int(away_score * 1.8 + rng_g.normal(0, 1.5)))
                home_walks = max(0, int(rng_g.normal(3.2, 1.2)))
                away_walks = max(0, int(rng_g.normal(3.2, 1.2)))
                home_k   = max(0, int(rng_g.normal(8.5, 2.0)))
                away_k   = max(0, int(rng_g.normal(8.5, 2.0)))
                home_ip  = float(np.clip(rng_g.normal(5.5, 1.0), 1.0, 9.0))
                away_ip  = float(np.clip(rng_g.normal(5.5, 1.0), 1.0, 9.0))

                home_sp_id = year * 1_000_000 + i * 10_000 + g * 100
                away_sp_id = year * 1_000_000 + j * 10_000 + g * 100 + 1

                games.append({
                    "date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "winner": home_team if home_score > away_score else away_team,
                    "run_diff": home_score - away_score,
                    "total_runs": home_score + away_score,
                    "home_runs_scored": float(home_score),
                    "home_runs_allowed_log": float(away_score),
                    "home_hits": float(home_hits),
                    "home_walks": float(home_walks),
                    "home_strikeouts": float(home_k),
                    "home_innings_pitched": float(home_ip),
                    "away_runs_scored": float(away_score),
                    "away_runs_allowed_log": float(home_score),
                    "away_hits": float(away_hits),
                    "away_walks": float(away_walks),
                    "away_strikeouts": float(away_k),
                    "away_innings_pitched": float(away_ip),
                    "home_sp_id": home_sp_id,
                    "home_sp_name": f"SP_{home_team}_{g}",
                    "away_sp_id": away_sp_id,
                    "away_sp_name": f"SP_{away_team}_{g}",
                    "home_lineup": _make_synthetic_lineup_json(home_team, year),
                    "away_lineup": _make_synthetic_lineup_json(away_team, year),
                    "season": year,
                    "temp_f": float(np.clip(rng_g.normal(68, 12), 45, 95)),
                    "wind_mph": float(np.clip(rng_g.normal(8, 5), 0, 25)),
                    "wind_out_factor": float(rng_g.uniform(-0.3, 0.3)),
                    "precip_prob": float(np.clip(rng_g.normal(0.1, 0.1), 0, 1)),
                    # v6 weather aliases (same values — avoid another API call)
                    "weather_temp_f": float(np.clip(rng_g.normal(68, 12), 45, 95)),
                    "weather_wind_mph": float(np.clip(rng_g.normal(8, 5), 0, 25)),
                    "weather_wind_out_factor": float(rng_g.uniform(-0.3, 0.3)),
                    "weather_precip": float(np.clip(rng_g.normal(0.1, 0.1), 0, 1)),
                })

    df = pd.DataFrame(games)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Generated synthetic season %d: %d games", year, len(df))
    print(f"  Synthetic {year}: {len(df)} games (offline fallback)")
    return df


# ---------------------------------------------------------------------------
# v2 helpers and fetch functions
# ---------------------------------------------------------------------------


def _safe_float(val, default=np.nan):
    """Safely convert a value to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Hardcoded MLB park coordinates (lat, lon) for weather lookups
PARK_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.8907, -84.4677),  "ARI": (33.4453, -112.0667),
    "BAL": (39.2838, -76.6215),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CHW": (41.8299, -87.6338),
    "CIN": (39.0978, -84.5082),  "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),  "KCR": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2777),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "OAK": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),  "PIT": (40.4469, -80.0057),
    "SDP": (32.7076, -117.1570), "SEA": (47.5914, -122.3325),
    "SFG": (37.7786, -122.3893), "STL": (38.6226, -90.1928),
    "TBR": (27.7682, -82.6534),  "TEX": (32.7473, -97.0845),
    "TOR": (43.6414, -79.3894),  "WSN": (38.8730, -77.0074),
}


def fetch_pitcher_season_stats(
    season: int, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch season-level pitching stats for all pitchers via MLB Stats API.

    Returns
    -------
    DataFrame indexed by pitcher_id with columns:
        pitcher_name, team, era, whip, k9, bb9, innings,
        wins, losses, games_started.
    Only includes pitchers with games_started >= 1.
    """
    cache_path = CACHE_DIR / f"pitcher_stats_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading pitcher stats from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching pitcher season stats for %d ...", season)

    all_splits: list = []
    limit = 500
    offset = 0
    while True:
        url = (
            f"{MLB_STATS_API}/stats"
            f"?stats=season&group=pitching&gameType=R"
            f"&season={season}&sportId=1"
            f"&playerPool=All&limit={limit}&offset={offset}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch pitcher stats for season %d: %s", season, exc)
            raise

        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            break

        all_splits.extend(splits)

        if len(splits) < limit:
            break

        offset += limit
        time.sleep(0.1)

    if not all_splits:
        raise ValueError(f"No pitcher stats returned for season {season}")

    splits = all_splits
    records: list[dict] = []

    for split in splits:
        try:
            player = split.get("player", {})
            team_info = split.get("team", {})
            stat = split.get("stat", {})

            pitcher_id = player.get("id")
            if pitcher_id is None:
                continue

            games_started = int(stat.get("gamesStarted", 0))
            if games_started < 1:
                continue

            strikeouts    = int(stat.get("strikeOuts", 0))
            batters_faced = int(stat.get("battersFaced", 0))
            air_outs_p    = int(stat.get("airOuts", 0))
            ground_outs_p = int(stat.get("groundOuts", 0))

            k_rate_p    = strikeouts / batters_faced if batters_faced > 0 else 0.0
            gb_fb_ratio = ground_outs_p / air_outs_p if air_outs_p > 0 else 1.0

            records.append({
                "pitcher_id": int(pitcher_id),
                "pitcher_name": player.get("fullName", ""),
                "team": _to_canonical(_mlb_team_name_to_abbrev(team_info.get("name", ""))),
                "era": _safe_float(stat.get("era")),
                "whip": _safe_float(stat.get("whip")),
                "k9": _safe_float(stat.get("strikeoutsPer9Inn")),
                "bb9": _safe_float(stat.get("walksPer9Inn")),
                "innings": _safe_float(stat.get("inningsPitched")),
                "wins": int(stat.get("wins", 0)),
                "losses": int(stat.get("losses", 0)),
                "games_started": games_started,
                "strikeouts":    strikeouts,
                "batters_faced": batters_faced,
                "hr_per9":       _safe_float(stat.get("homeRunsPer9")),
                "hits_per9":     _safe_float(stat.get("hitsPer9Inn")),
                "air_outs":      air_outs_p,
                "ground_outs":   ground_outs_p,
                "k_rate":        k_rate_p,
                "gb_fb_ratio":   gb_fb_ratio,
            })
        except Exception as exc:
            logger.debug("Skipping pitcher split: %s", exc)
            continue

    if not records:
        raise ValueError(f"No pitcher data parsed for season {season}")

    df = pd.DataFrame(records).set_index("pitcher_id")
    df.to_parquet(cache_path)
    logger.info("Cached pitcher stats: %d pitchers for season %d", len(df), season)
    return df


ROTOWIRE_TO_BR: dict[str, str] = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL",
    "Boston": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati": "CIN", "Cleveland": "CLE", "Colorado": "COL",
    "Detroit": "DET", "Houston": "HOU", "Kansas City": "KCR",
    "LA Angels": "LAA", "LA Dodgers": "LAD", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "NY Mets": "NYM",
    "NY Yankees": "NYY", "Oakland": "OAK", "Philadelphia": "PHI",
    "Pittsburgh": "PIT", "San Diego": "SDP", "Seattle": "SEA",
    "San Francisco": "SFG", "St. Louis": "STL", "Tampa Bay": "TBR",
    "Texas": "TEX", "Toronto": "TOR", "Washington": "WSN",
}

# Map MLB Stats API team name -> abbreviation
_MLB_TEAM_NAME_TO_ABBREV: dict[str, str] = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP',
    'Seattle Mariners': 'SEA', 'San Francisco Giants': 'SFG',
    'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN',
}

def _mlb_team_name_to_abbrev(name: str) -> str:
    return _MLB_TEAM_NAME_TO_ABBREV.get(name, '')

_PROBABLE_EMPTY_COLS = [
    "game_pk", "home_team", "away_team",
    "home_pitcher_id", "home_pitcher_name",
    "away_pitcher_id", "away_pitcher_name",
]


def _fetch_probable_pitchers_mlb(
    game_date, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch probable starters from MLB Stats API for all games on a given date.

    Cache expires after 6 hours (probable pitchers change day-of).

    Returns
    -------
    DataFrame with columns: game_pk, home_team, away_team,
        home_pitcher_id, home_pitcher_name, away_pitcher_id, away_pitcher_name.
    """
    if hasattr(game_date, "strftime"):
        date_str = game_date.strftime("%Y-%m-%d")
    else:
        date_str = str(game_date)

    cache_path = CACHE_DIR / f"probable_pitchers_{date_str}.parquet"

    # Use cache if less than 6 hours old
    if not force_refresh and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < 6:
            logger.debug("Loading probable pitchers from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    logger.info("Fetching probable pitchers for %s ...", date_str)

    url = (
        f"{MLB_STATS_API}/schedule"
        f"?sportId=1&date={date_str}&gameType=R"
        f"&hydrate=probablePitcher(note),team"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch probable pitchers for %s: %s", date_str, exc)
        raise

    records: list[dict] = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            try:
                home_info = game["teams"]["home"]
                away_info = game["teams"]["away"]
                records.append({
                    "game_pk": game.get("gamePk"),
                    "home_team": _to_canonical(home_info["team"]["abbreviation"]),
                    "away_team": _to_canonical(away_info["team"]["abbreviation"]),
                    "home_pitcher_id": home_info.get("probablePitcher", {}).get("id"),
                    "home_pitcher_name": home_info.get("probablePitcher", {}).get("fullName"),
                    "away_pitcher_id": away_info.get("probablePitcher", {}).get("id"),
                    "away_pitcher_name": away_info.get("probablePitcher", {}).get("fullName"),
                })
            except Exception as exc:
                logger.debug("Skipping game in probable pitchers fetch: %s", exc)
                continue

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=_PROBABLE_EMPTY_COLS)
    df.to_parquet(cache_path, index=False)
    logger.info("Probable pitchers (MLB API) for %s: %d games", date_str, len(df))
    return df


def fetch_probable_pitchers_rotowire(game_date) -> pd.DataFrame:
    """Scrape probable starters from Rotowire.

    URL: https://www.rotowire.com/baseball/daily-lineups.php

    Returns same schema as fetch_probable_pitchers():
        game_pk, home_team, away_team,
        home_pitcher_id, home_pitcher_name,
        away_pitcher_id, away_pitcher_name

    pitcher_id will be None (Rotowire doesn't have MLB IDs).
    pitcher_name will be the full name string.

    Cache path: CACHE_DIR / f"probable_pitchers_rotowire_{game_date}.parquet"
    Cache expiry: 6 hours (same as MLB Stats API version)
    """
    from bs4 import BeautifulSoup

    if hasattr(game_date, "strftime"):
        date_str = game_date.strftime("%Y-%m-%d")
    else:
        date_str = str(game_date)

    cache_path = CACHE_DIR / f"probable_pitchers_rotowire_{date_str}.parquet"

    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < 6:
            logger.debug("Loading Rotowire probable pitchers from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    logger.info("Scraping Rotowire probable pitchers for %s ...", date_str)

    url = f"https://www.rotowire.com/baseball/daily-lineups.php?date={date_str}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        time.sleep(1)
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Rotowire fetch failed for %s: %s", date_str, exc)
        return pd.DataFrame(columns=_PROBABLE_EMPTY_COLS)

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        game_boxes = soup.find_all("div", class_="lineup__box")

        records: list[dict] = []
        for box in game_boxes:
            try:
                # Team abbreviations are in lineup__abbr divs inside is-visit / is-home
                visit_team_el = box.find("div", class_="lineup__team is-visit")
                home_team_el = box.find("div", class_="lineup__team is-home")
                if not visit_team_el or not home_team_el:
                    continue

                away_abbr_raw = visit_team_el.find("div", class_="lineup__abbr")
                home_abbr_raw = home_team_el.find("div", class_="lineup__abbr")
                if not away_abbr_raw or not home_abbr_raw:
                    continue

                away_abbr = _to_canonical(away_abbr_raw.get_text(strip=True))
                home_abbr = _to_canonical(home_abbr_raw.get_text(strip=True))

                # Validate game date from the matchup href (e.g. /baseball/box-score/...-2026-03-26-12345)
                matchup_link = box.find("a", class_="lineup__matchup")
                href = matchup_link.get("href", "") if matchup_link else ""
                href_date_m = re.search(r"(\d{4}-\d{2}-\d{2})", href)
                if href_date_m:
                    href_date = href_date_m.group(1)
                    if href_date != date_str:
                        logger.debug(
                            "Skipping Rotowire box %s @ %s for date %s (requested %s)",
                            away_abbr, home_abbr, href_date, date_str,
                        )
                        continue

                # Extract probable pitcher from the highlighted player slot in each list
                def _get_pitcher(ul_el):
                    if ul_el is None:
                        return None
                    hi = ul_el.find("li", class_="lineup__player-highlight")
                    if hi is None:
                        return None
                    name_div = hi.find("div", class_="lineup__player-highlight-name")
                    if name_div is None:
                        return None
                    a_tag = name_div.find("a")
                    name = a_tag.get_text(strip=True) if a_tag else name_div.get_text(strip=True)
                    return name if name and name.lower() not in ("tbd", "") else None

                visit_list = box.find("ul", class_="lineup__list is-visit")
                home_list = box.find("ul", class_="lineup__list is-home")

                records.append({
                    "game_pk": None,
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "home_pitcher_id": None,
                    "home_pitcher_name": _get_pitcher(home_list),
                    "away_pitcher_id": None,
                    "away_pitcher_name": _get_pitcher(visit_list),
                })
            except Exception as exc:
                logger.debug("Skipping Rotowire game box: %s", exc)
                continue

    except Exception as exc:
        logger.warning("Rotowire parse failed for %s: %s", date_str, exc)
        return pd.DataFrame(columns=_PROBABLE_EMPTY_COLS)

    if not records:
        logger.info(
            "Rotowire returned no valid games for %s (wrong-date or no data); "
            "falling back to MLB Stats API.",
            date_str,
        )
        return _fetch_probable_pitchers_mlb(game_date)

    df = pd.DataFrame(records)
    df.to_parquet(cache_path, index=False)
    logger.info("Probable pitchers (Rotowire) for %s: %d games", date_str, len(df))
    return df


def fetch_probable_pitchers(
    game_date, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch probable starters with fallback chain.

    1. MLB Stats API (primary)
    2. Rotowire (fallback if MLB API returns empty or incomplete)

    Merges results: MLB API provides pitcher_id, Rotowire provides
    reliable names. If MLB API has the game but no pitcher_id,
    tries to match Rotowire name → pitcher_id via pitcher_stats lookup.

    Returns
    -------
    DataFrame with columns: game_pk, home_team, away_team,
        home_pitcher_id, home_pitcher_name, away_pitcher_id, away_pitcher_name.
    """
    # Try MLB Stats API first
    mlb_df = _fetch_probable_pitchers_mlb(game_date, force_refresh)

    # If MLB API returned pitchers for 80%+ of games, use it as-is
    if not mlb_df.empty:
        games_with_pitchers = mlb_df[
            mlb_df["home_pitcher_name"].notna() &
            mlb_df["away_pitcher_name"].notna()
        ]
        if len(games_with_pitchers) >= len(mlb_df) * 0.8:
            return mlb_df

    missing = len(mlb_df) - (
        int(mlb_df["home_pitcher_name"].notna().sum()) if not mlb_df.empty else 0
    )
    print(
        f"MLB API missing pitchers for {missing} games, trying Rotowire..."
    )

    roto_df = fetch_probable_pitchers_rotowire(game_date)
    if roto_df.empty:
        return mlb_df  # return MLB data even if incomplete

    # Validate Rotowire games against MLB API schedule; drop any that don't match
    if not mlb_df.empty and not roto_df.empty:
        mlb_pairs = set(zip(mlb_df["home_team"], mlb_df["away_team"]))
        valid_mask = roto_df.apply(
            lambda r: (r["home_team"], r["away_team"]) in mlb_pairs, axis=1
        )
        for _, row in roto_df[~valid_mask].iterrows():
            logger.warning(
                "Rotowire game %s @ %s not found in MLB schedule — dropping.",
                row["away_team"], row["home_team"],
            )
        roto_df = roto_df[valid_mask].reset_index(drop=True)
        if roto_df.empty:
            return mlb_df

    # Build name → id lookup from pitcher_season_stats
    try:
        from datetime import date as _date
        game_date_obj = game_date if hasattr(game_date, "year") else _date.fromisoformat(str(game_date))
        pitcher_stats = fetch_pitcher_season_stats(game_date_obj.year)
        name_to_id = dict(zip(pitcher_stats["pitcher_name"], pitcher_stats.index))
    except Exception:
        name_to_id = {}

    # For each Rotowire row, try to resolve pitcher IDs
    for idx, row in roto_df.iterrows():
        for side in ["home", "away"]:
            name = row.get(f"{side}_pitcher_name")
            if name and pd.isna(row.get(f"{side}_pitcher_id")):
                pitcher_id = name_to_id.get(name)
                roto_df.at[idx, f"{side}_pitcher_id"] = pitcher_id

    if mlb_df.empty:
        return roto_df[_PROBABLE_EMPTY_COLS]

    # Merge MLB and Rotowire: prefer MLB where available, fill gaps with Rotowire
    merged = mlb_df.merge(
        roto_df[["home_team", "away_team",
                 "home_pitcher_name", "away_pitcher_name",
                 "home_pitcher_id", "away_pitcher_id"]],
        on=["home_team", "away_team"],
        how="left",
        suffixes=("_mlb", "_roto"),
    )

    for side in ["home", "away"]:
        merged[f"{side}_pitcher_name"] = merged[f"{side}_pitcher_name_mlb"].fillna(
            merged[f"{side}_pitcher_name_roto"]
        )
        merged[f"{side}_pitcher_id"] = merged[f"{side}_pitcher_id_mlb"].fillna(
            merged[f"{side}_pitcher_id_roto"]
        )

    return merged[_PROBABLE_EMPTY_COLS]


def fetch_bullpen_stats(
    season: int, force_refresh: bool = False
) -> pd.DataFrame:
    """Compute rolling team pitching ERA proxy per team from cached game logs.

    Uses shift(1) before rolling to prevent future leakage.

    Returns
    -------
    DataFrame with columns: team, date, era_r7, era_r15.
    """
    cache_path = CACHE_DIR / f"bullpen_stats_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading bullpen stats from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Computing bullpen stats for %d from game logs ...", season)

    logs = fetch_team_game_logs(season, force_refresh=force_refresh)
    if logs.empty:
        raise ValueError(f"No team game logs for season {season}")

    logs = logs.copy()
    logs["runs_allowed"] = pd.to_numeric(logs["runs_allowed"], errors="coerce")
    logs["innings_pitched"] = pd.to_numeric(logs["innings_pitched"], errors="coerce")

    result_parts: list[pd.DataFrame] = []

    for _team, group in logs.groupby("team"):
        group = group.sort_values("date").copy()

        # Shift 1 to prevent leakage
        s_runs = group["runs_allowed"].shift(1)
        s_innings = group["innings_pitched"].shift(1)

        runs_r7 = s_runs.rolling(7, min_periods=3).sum()
        inn_r7 = s_innings.rolling(7, min_periods=3).sum().replace(0, np.nan)
        runs_r15 = s_runs.rolling(15, min_periods=5).sum()
        inn_r15 = s_innings.rolling(15, min_periods=5).sum().replace(0, np.nan)

        group["era_r7"] = (runs_r7 * 9) / inn_r7
        group["era_r15"] = (runs_r15 * 9) / inn_r15

        result_parts.append(group[["team", "date", "era_r7", "era_r15"]])

    df = pd.concat(result_parts, ignore_index=True)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached bullpen stats: %d rows for season %d", len(df), season)
    return df


def fetch_game_starters(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch the actual starting pitcher for every completed regular-season
    game in a season using the MLB Stats API box score endpoint.
    Returns DataFrame with columns:
      game_pk, date, home_team, away_team,
      home_sp_id, home_sp_name,
      away_sp_id, away_sp_name
    Cache: CACHE_DIR / f"game_starters_{season}.parquet"
    """
    cache_path = CACHE_DIR / f"game_starters_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading game starters from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching game starters for season %d ...", season)
    t0 = time.time()

    # Step 1: Get all completed regular-season games from schedule
    url = (
        f"{MLB_STATS_API}/schedule"
        f"?sportId=1&season={season}&gameType=R&hydrate=team"
        f"&fields=dates,date,games,gamePk,teams,home,away,team,"
        f"abbreviation,score,status,abstractGameState"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch schedule for game starters %d: %s", season, exc)
        raise

    game_pks = []
    for date_entry in data.get("dates", []):
        date_str = date_entry.get("date", "")
        try:
            game_date = pd.to_datetime(date_str)
        except Exception:
            continue
        # Skip spring training (before April 1)
        if game_date.month < 4:
            continue
        for game in date_entry.get("games", []):
            try:
                state = game.get("status", {}).get("abstractGameState", "")
                if state != "Final":
                    continue
                game_pk = game.get("gamePk")
                if not game_pk:
                    continue
                home_abbrev = game["teams"]["home"]["team"]["abbreviation"]
                away_abbrev = game["teams"]["away"]["team"]["abbreviation"]
                game_pks.append({
                    "game_pk": game_pk,
                    "date": date_str,
                    "home_team": _to_canonical(home_abbrev),
                    "away_team": _to_canonical(away_abbrev),
                })
            except Exception as exc:
                logger.debug("Skipping game in starters fetch: %s", exc)
                continue

    total = len(game_pks)
    logger.info("Found %d completed games for season %d", total, season)

    # Step 2: Fetch boxscore for each game to get actual starting pitchers and lineups
    rows = []
    for i, game_info in enumerate(game_pks, 1):
        if i % 100 == 0:
            print(f"  Box scores: {i}/{total} games fetched...")
        game_pk = game_info["game_pk"]
        try:
            box_url = f"{MLB_STATS_API}/game/{game_pk}/boxscore"
            box_resp = requests.get(box_url, timeout=30)
            box_resp.raise_for_status()
            box = box_resp.json()

            home_pitchers = box["teams"]["home"]["pitchers"]
            away_pitchers = box["teams"]["away"]["pitchers"]
            home_sp_id = home_pitchers[0] if home_pitchers else None
            away_sp_id = away_pitchers[0] if away_pitchers else None

            players = {}
            players.update(box["teams"]["home"].get("players", {}))
            players.update(box["teams"]["away"].get("players", {}))

            home_sp_name = None
            if home_sp_id:
                home_sp_name = (
                    players.get(f"ID{home_sp_id}", {})
                    .get("person", {})
                    .get("fullName")
                )

            away_sp_name = None
            if away_sp_id:
                away_sp_name = (
                    players.get(f"ID{away_sp_id}", {})
                    .get("person", {})
                    .get("fullName")
                )

            # Extract batting lineups
            lineup_jsons = {}
            for side in ["home", "away"]:
                batters = box["teams"][side].get("batters", [])
                side_players = box["teams"][side].get("players", {})
                lineup = []
                for order, player_id in enumerate(batters[:9], 1):
                    key = f"ID{player_id}"
                    player = side_players.get(key, {})
                    name = player.get("person", {}).get("fullName", "")
                    pos = player.get("position", {}).get("abbreviation", "")
                    lineup.append({
                        "id": player_id,
                        "name": name,
                        "batting_order": order,
                        "position": pos,
                    })
                lineup_jsons[side] = json.dumps(lineup)

            rows.append({
                "game_pk": game_pk,
                "date": game_info["date"],
                "home_team": game_info["home_team"],
                "away_team": game_info["away_team"],
                "home_sp_id": home_sp_id,
                "home_sp_name": home_sp_name,
                "away_sp_id": away_sp_id,
                "away_sp_name": away_sp_name,
                "home_lineup": lineup_jsons["home"],
                "away_lineup": lineup_jsons["away"],
            })
        except Exception as exc:
            logger.debug("Skipping boxscore for game_pk %s: %s", game_pk, exc)
            continue
        time.sleep(0.05)

    logger.info(
        "fetch_game_starters(%d) complete: %d games in %.1fs",
        season, len(rows), time.time() - t0,
    )

    if not rows:
        df = pd.DataFrame(columns=[
            "game_pk", "date", "home_team", "away_team",
            "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
            "home_lineup", "away_lineup",
        ])
    else:
        df = pd.DataFrame(rows)

    df.to_parquet(cache_path, index=False)
    return df


def fetch_pitcher_game_logs(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch game-by-game pitching appearance logs for all pitchers
    in a season.
    Returns one row per pitcher appearance:
      pitcher_id, pitcher_name, date, season,
      innings_pitched, earned_runs, hits_allowed,
      walks_allowed, strikeouts, game_pk
    Cache: CACHE_DIR / f"pitcher_game_logs_{season}.parquet"
    """
    cache_path = CACHE_DIR / f"pitcher_game_logs_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading pitcher game logs from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching pitcher game logs for season %d ...", season)
    t0 = time.time()

    # Get all pitcher IDs for the season via pitcher season stats
    try:
        pitcher_stats = fetch_pitcher_season_stats(season, force_refresh=force_refresh)
    except Exception as exc:
        logger.error("Failed to fetch pitcher season stats for %d: %s", season, exc)
        empty_cols = [
            "pitcher_id", "pitcher_name", "date", "season",
            "innings_pitched", "earned_runs", "hits_allowed",
            "walks_allowed", "strikeouts", "game_pk",
        ]
        df = pd.DataFrame(columns=empty_cols)
        df.to_parquet(cache_path, index=False)
        return df

    pitcher_ids = pitcher_stats.index.tolist()
    pitcher_names = pitcher_stats["pitcher_name"].tolist()
    total = len(pitcher_ids)

    all_rows: list[dict] = []

    for i, (pitcher_id, pitcher_name) in enumerate(zip(pitcher_ids, pitcher_names), 1):
        if i % 50 == 0:
            print(f"  Pitcher logs: {i}/{total} ({pitcher_name})...")
        try:
            url = (
                f"{MLB_STATS_API}/people/{pitcher_id}/stats"
                f"?stats=gameLog&group=pitching&season={season}&gameType=R"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for stat_group in data.get("stats", []):
                for split in stat_group.get("splits", []):
                    try:
                        stat = split.get("stat", {})
                        all_rows.append({
                            "pitcher_id": int(pitcher_id),
                            "pitcher_name": pitcher_name,
                            "date": split["date"],
                            "season": season,
                            "innings_pitched": _safe_float(stat.get("inningsPitched")),
                            "earned_runs": _safe_float(stat.get("earnedRuns")),
                            "hits_allowed": _safe_float(stat.get("hits")),
                            "walks_allowed": _safe_float(stat.get("baseOnBalls")),
                            "strikeouts": _safe_float(stat.get("strikeOuts")),
                            "game_pk": split.get("game", {}).get("gamePk"),
                        })
                    except Exception as exc:
                        logger.debug(
                            "Skipping split for pitcher %s: %s", pitcher_id, exc
                        )
                        continue
        except Exception as exc:
            logger.debug("Skipping pitcher %s (%s): %s", pitcher_id, pitcher_name, exc)
            continue
        time.sleep(0.05)

    logger.info(
        "fetch_pitcher_game_logs(%d) complete: %d rows, %d pitchers in %.1fs",
        season, len(all_rows), total, time.time() - t0,
    )

    if not all_rows:
        empty_cols = [
            "pitcher_id", "pitcher_name", "date", "season",
            "innings_pitched", "earned_runs", "hits_allowed",
            "walks_allowed", "strikeouts", "game_pk",
        ]
        df = pd.DataFrame(columns=empty_cols)
    else:
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"])

    df.to_parquet(cache_path, index=False)
    return df


def fetch_batter_season_stats(
    season: int, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch season-level batting stats for all batters via MLB Stats API.

    Used to look up OPS and wRC+ proxy for lineup strength.

    Returns
    -------
    DataFrame indexed by batter_id with columns:
        batter_name, team, ops, obp, slg, plate_appearances,
        home_runs, wrc_plus_proxy.
    Only includes batters with plate_appearances >= 50.
    """
    cache_path = CACHE_DIR / f"batter_stats_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading batter stats from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching batter season stats for %d ...", season)

    all_splits: list = []
    limit = 500
    offset = 0
    while True:
        url = (
            f"{MLB_STATS_API}/stats"
            f"?stats=season&group=hitting&gameType=R"
            f"&season={season}&sportId=1"
            f"&playerPool=All&limit={limit}&offset={offset}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "fetch_batter_season_stats(%d) failed (%s) — using synthetic stats.",
                season, exc,
            )
            return generate_synthetic_batter_stats(season)

        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            break

        all_splits.extend(splits)

        if len(splits) < limit:
            break

        offset += limit
        time.sleep(0.1)

    if not all_splits:
        raise ValueError(f"No batter stats returned for season {season}")

    records: list[dict] = []

    for split in all_splits:
        try:
            player = split.get("player", {})
            team_info = split.get("team", {})
            stat = split.get("stat", {})

            batter_id = player.get("id")
            if batter_id is None:
                continue

            plate_appearances = int(stat.get("plateAppearances", 0))
            if plate_appearances < 50:
                continue

            obp = _safe_float(stat.get("obp"))
            slg = _safe_float(stat.get("slg"))
            ops = _safe_float(stat.get("ops"))

            # wRC+ proxy: (OBP + SLG) / 0.720 * 100
            if not np.isnan(obp) and not np.isnan(slg):
                wrc_plus_proxy = (obp + slg) / 0.720 * 100
            else:
                wrc_plus_proxy = np.nan

            home_runs   = int(stat.get("homeRuns", 0))
            at_bats     = int(stat.get("atBats", 0))
            strikeouts  = int(stat.get("strikeOuts", 0))
            walks       = int(stat.get("baseOnBalls", 0))
            total_bases = int(stat.get("totalBases", 0))
            air_outs    = int(stat.get("airOuts", 0))

            hr_rate      = home_runs / plate_appearances if plate_appearances > 0 else 0.0
            k_rate       = strikeouts / plate_appearances if plate_appearances > 0 else 0.0
            bb_rate      = walks / plate_appearances if plate_appearances > 0 else 0.0
            tb_rate      = total_bases / plate_appearances if plate_appearances > 0 else 0.0
            hr_per_air   = home_runs / air_outs if air_outs > 0 else 0.0

            records.append({
                "batter_id": int(batter_id),
                "batter_name": player.get("fullName", ""),
                "team": _to_canonical(_mlb_team_name_to_abbrev(team_info.get("name", ""))),
                "ops": ops,
                "obp": obp,
                "slg": slg,
                "plate_appearances": plate_appearances,
                "home_runs": home_runs,
                "wrc_plus_proxy": wrc_plus_proxy,
                "at_bats":      at_bats,
                "strikeouts":   strikeouts,
                "walks":        walks,
                "total_bases":  total_bases,
                "ab_per_hr":    _safe_float(stat.get("atBatsPerHomeRun")),
                "air_outs":     air_outs,
                "babip":        _safe_float(stat.get("babip")),
                "doubles":      int(stat.get("doubles", 0)),
                "triples":      int(stat.get("triples", 0)),
                "hr_rate":      hr_rate,
                "k_rate":       k_rate,
                "bb_rate":      bb_rate,
                "tb_rate":      tb_rate,
                "hr_per_air":   hr_per_air,
            })
        except Exception as exc:
            logger.debug("Skipping batter split: %s", exc)
            continue

    if not records:
        raise ValueError(f"No batter data parsed for season {season}")

    df = pd.DataFrame(records).set_index("batter_id")
    df.to_parquet(cache_path)
    logger.info("Cached batter stats: %d batters for season %d", len(df), season)
    print(f"Total batters fetched for {season}: {len(df)}")
    return df


def fetch_batter_game_logs(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch game-by-game batting logs for regular starters.

    Only fetches batters with PA >= 200 to limit API calls (~150 batters).

    Returns
    -------
    DataFrame with columns:
        batter_id, batter_name, date, game_pk,
        at_bats, hits, walks, home_runs, plate_appearances.
    Cache: CACHE_DIR / f"batter_game_logs_{season}.parquet"
    """
    cache_path = CACHE_DIR / f"batter_game_logs_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading batter game logs from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching batter game logs for season %d ...", season)
    t0 = time.time()

    # Get batters with PA >= 200
    try:
        batter_stats = fetch_batter_season_stats(season, force_refresh=force_refresh)
    except Exception as exc:
        logger.error("Failed to fetch batter season stats for %d: %s", season, exc)
        empty_cols = [
            "batter_id", "batter_name", "date", "game_pk",
            "at_bats", "hits", "walks", "home_runs", "plate_appearances",
        ]
        df = pd.DataFrame(columns=empty_cols)
        df.to_parquet(cache_path, index=False)
        return df

    regulars = batter_stats[batter_stats["plate_appearances"] >= 200]
    batter_ids = regulars.index.tolist()
    batter_names = regulars["batter_name"].tolist()
    total = len(batter_ids)

    # Synthetic player IDs are season * 10_000 + ... ≥ 20_220_000 — far above
    # any real MLB player ID (typically 5-7 digits).  Skip API calls entirely.
    if batter_ids and batter_ids[0] > 10_000_000:
        logger.info("Synthetic batter IDs detected for %d — skipping game log fetch.", season)
        empty_cols = [
            "batter_id", "batter_name", "date", "game_pk",
            "at_bats", "hits", "walks", "home_runs", "plate_appearances",
        ]
        df = pd.DataFrame(columns=empty_cols)
        df.to_parquet(cache_path, index=False)
        return df

    all_rows: list[dict] = []

    for i, (batter_id, batter_name) in enumerate(zip(batter_ids, batter_names), 1):
        if i % 25 == 0:
            print(f"  Batter logs: {i}/{total} ({batter_name})...")
        try:
            url = (
                f"{MLB_STATS_API}/people/{batter_id}/stats"
                f"?stats=gameLog&group=hitting&season={season}&gameType=R"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for stat_group in data.get("stats", []):
                for split in stat_group.get("splits", []):
                    try:
                        stat = split.get("stat", {})
                        all_rows.append({
                            "batter_id": int(batter_id),
                            "batter_name": batter_name,
                            "date": split["date"],
                            "game_pk": split.get("game", {}).get("gamePk"),
                            "at_bats": _safe_float(stat.get("atBats")),
                            "hits": _safe_float(stat.get("hits")),
                            "walks": _safe_float(stat.get("baseOnBalls")),
                            "home_runs": _safe_float(stat.get("homeRuns")),
                            "plate_appearances": _safe_float(stat.get("plateAppearances")),
                        })
                    except Exception as exc:
                        logger.debug(
                            "Skipping split for batter %s: %s", batter_id, exc
                        )
                        continue
        except Exception as exc:
            logger.debug("Skipping batter %s (%s): %s", batter_id, batter_name, exc)
            continue
        time.sleep(0.05)

    logger.info(
        "fetch_batter_game_logs(%d) complete: %d rows, %d batters in %.1fs",
        season, len(all_rows), total, time.time() - t0,
    )

    if not all_rows:
        empty_cols = [
            "batter_id", "batter_name", "date", "game_pk",
            "at_bats", "hits", "walks", "home_runs", "plate_appearances",
        ]
        df = pd.DataFrame(columns=empty_cols)
    else:
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"])

    df.to_parquet(cache_path, index=False)
    return df


def fetch_historical_weather(
    season: int,
    force_refresh: bool = False,
    max_games: int = 50,
    allow_full_fetch: bool = False,
) -> pd.DataFrame:
    """Fetch historical game-time weather for every game in a season.

    Uses Open-Meteo historical archive API:
      https://archive-api.open-meteo.com/v1/archive

    Returns DataFrame with columns:
      game_id, date, home_team,
      temp_f, wind_mph, wind_dir,
      precip_prob, wind_out_factor

    Cache path:
      CACHE_DIR / f"historical_weather_{season}.parquet"

    Parameters
    ----------
    max_games:
        If the number of unique park-dates that need fetching exceeds this
        limit, skip the network fetch entirely and return whatever is already
        cached (or an empty DataFrame if no cache exists).  Prevents runaway
        fetches for full seasons with 2,400+ games.
    allow_full_fetch:
        If True, raise the effective limit to 200 park-dates.  Set this to
        True for the current season where early-season schedules have far
        fewer games than a completed 162-game season (2,400+ park-dates).
    """
    import datetime as _dt
    current_year = _dt.date.today().year
    effective_max = 200 if allow_full_fetch or season == current_year else max_games

    cache_path = CACHE_DIR / f"historical_weather_{season}.parquet"
    if not force_refresh and cache_path.exists():
        logger.debug("Loading historical weather from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    # The archive-api.open-meteo.com endpoint only serves completed historical
    # dates and returns 400 errors for any date in the current (in-progress)
    # season.  Skip the fetch entirely and return empty; callers that need
    # current-season weather should use the TTL-cached live fetch_weather().
    if season == current_year:
        logger.info(
            "Skipping historical weather fetch for current season %d "
            "— use live fetch_weather() instead",
            season,
        )
        return pd.DataFrame(
            columns=[
                "game_id", "date", "home_team",
                "temp_f", "wind_mph", "wind_dir", "precip_prob", "wind_out_factor",
            ]
        )

    logger.info("Fetching historical weather for season %d ...", season)
    t0 = time.time()

    schedule = fetch_schedule_and_results(season, force_refresh=False)
    if schedule.empty:
        logger.warning(
            "No schedule data for season %d; returning empty weather DataFrame", season
        )
        return pd.DataFrame(
            columns=[
                "game_id", "date", "home_team",
                "temp_f", "wind_mph", "wind_dir", "precip_prob", "wind_out_factor",
            ]
        )

    # Group by (home_team, date) so each unique park-date is fetched once
    park_dates = schedule[["home_team", "date"]].drop_duplicates().copy()
    park_dates["date_str"] = park_dates["date"].dt.strftime("%Y-%m-%d")

    # Guard: if there are too many park-dates to fetch, return existing cache or empty.
    # Current-season fetches use effective_max=200; completed seasons use max_games=50.
    if len(park_dates) > effective_max:
        if cache_path.exists():
            logger.warning(
                "Season %d has %d park-dates to fetch (limit %d); "
                "returning existing cache as-is.",
                season, len(park_dates), effective_max,
            )
            return pd.read_parquet(cache_path)
        logger.warning(
            "Season %d has %d park-dates to fetch (limit %d); "
            "skipping fetch entirely and returning empty DataFrame.",
            season, len(park_dates), effective_max,
        )
        return pd.DataFrame(
            columns=[
                "game_id", "date", "home_team",
                "temp_f", "wind_mph", "wind_dir", "precip_prob", "wind_out_factor",
            ]
        )

    weather_records: dict[tuple, dict | None] = {}

    for i, (_, row) in enumerate(park_dates.iterrows()):
        home_team = str(row["home_team"])
        date_str = str(row["date_str"])

        if i > 0 and i % 200 == 0:
            logger.info(
                "Historical weather: processed %d / %d park-dates", i, len(park_dates)
            )

        coords = PARK_COORDS.get(home_team)
        if coords is None:
            logger.debug("No park coordinates for team %s; skipping", home_team)
            weather_records[(home_team, date_str)] = None
            time.sleep(0.1)
            continue

        lat, lon = coords
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            "&hourly=temperature_2m,windspeed_10m,winddirection_10m,precipitation"
            "&temperature_unit=fahrenheit"
            "&windspeed_unit=mph"
            "&timezone=America/New_York"
        )

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "Weather archive fetch failed for %s on %s: %s", home_team, date_str, exc
            )
            weather_records[(home_team, date_str)] = None
            time.sleep(0.1)
            continue

        try:
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            # Find index for 7pm (19:00) local time; fall back to nearest hour
            target_hour = f"{date_str}T19:00"
            if target_hour in times:
                idx = times.index(target_hour)
            elif times:
                def _hour_dist(t: str) -> float:
                    try:
                        return abs(int(t.split("T")[1].split(":")[0]) - 19)
                    except Exception:
                        return 99.0
                idx = min(range(len(times)), key=lambda ii: _hour_dist(times[ii]))
            else:
                weather_records[(home_team, date_str)] = None
                time.sleep(0.1)
                continue

            temp_f = float(hourly["temperature_2m"][idx])
            wind_mph = float(hourly["windspeed_10m"][idx])
            wind_dir = float(hourly["winddirection_10m"][idx])
            precip_mm = float(hourly["precipitation"][idx])
            precip_prob = min(precip_mm / 10.0, 1.0)

            wind_dir_rad = math.radians(wind_dir)
            if home_team == "COL":
                wind_out_factor = abs(wind_mph * math.cos(wind_dir_rad) * 0.1)
            else:
                wind_out_factor = wind_mph * math.cos(wind_dir_rad) * 0.1

            weather_records[(home_team, date_str)] = {
                "temp_f": temp_f,
                "wind_mph": wind_mph,
                "wind_dir": wind_dir,
                "precip_prob": precip_prob,
                "wind_out_factor": wind_out_factor,
            }
        except Exception as exc:
            logger.warning(
                "Weather archive parse failed for %s on %s: %s", home_team, date_str, exc
            )
            weather_records[(home_team, date_str)] = None

        time.sleep(0.1)

    # Build weather DataFrame aligned to schedule games
    out_rows = []
    for _, srow in schedule.iterrows():
        home_team = str(srow["home_team"])
        date_str = srow["date"].strftime("%Y-%m-%d")
        w = weather_records.get((home_team, date_str))
        rec: dict = {
            "game_id": srow["game_id"],
            "date": srow["date"],
            "home_team": home_team,
        }
        if w is not None:
            rec.update(w)
        else:
            rec.update({
                "temp_f": np.nan,
                "wind_mph": np.nan,
                "wind_dir": np.nan,
                "precip_prob": np.nan,
                "wind_out_factor": np.nan,
            })
        out_rows.append(rec)

    df = pd.DataFrame(out_rows)
    df.to_parquet(cache_path, index=False)

    elapsed = time.time() - t0
    logger.info(
        "Historical weather for season %d: %d rows, %.1fs elapsed",
        season, len(df), elapsed,
    )
    return df


# Module-level TTL cache: (team, date_str) → {"data": dict|None, "ts": float}
# Entries expire after 30 minutes to avoid hammering the free-tier Open-Meteo API
# (rate limit ~10 req/min) when predict_mlb_games is called repeatedly.
_weather_cache: dict[tuple, dict] = {}

_WEATHER_CACHE_TTL = 1800  # seconds


def fetch_weather(home_team: str, game_date) -> dict | None:
    """Fetch game-time weather via Open-Meteo (no API key required).

    Results are cached in memory for 30 minutes keyed by (team, date) to
    avoid 429 rate-limit errors when many games are predicted in one run.

    Returns dict with keys: temp_f, wind_mph, wind_dir, precip_prob,
    wind_out_factor.  Returns None on any failure.
    """
    coords = PARK_COORDS.get(home_team)
    if coords is None:
        logger.warning("No park coordinates for team %s", home_team)
        return None

    if hasattr(game_date, "strftime"):
        date_str = game_date.strftime("%Y-%m-%d")
    else:
        date_str = str(game_date)

    # Check TTL cache before making an HTTP request
    cache_key = (home_team, date_str)
    cached = _weather_cache.get(cache_key)
    if cached is not None and time.time() - cached["ts"] < _WEATHER_CACHE_TTL:
        logger.debug("Weather cache hit for %s on %s", home_team, date_str)
        return cached["data"]

    lat, lon = coords
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability"
        "&temperature_unit=fahrenheit"
        "&windspeed_unit=mph"
        "&timezone=America/New_York"
        f"&start_date={date_str}&end_date={date_str}"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Weather fetch failed for %s on %s: %s", home_team, date_str, exc)
        _weather_cache[cache_key] = {"data": None, "ts": time.time()}
        return None

    try:
        hourly = data.get("hourly", {})
        idx = 19  # closest to 7 pm local time
        temp_f = float(hourly["temperature_2m"][idx])
        wind_mph = float(hourly["windspeed_10m"][idx])
        wind_dir = float(hourly["winddirection_10m"][idx])
        precip_prob = float(hourly["precipitation_probability"][idx])

        # wind_out_factor: positive = wind blowing out = more runs
        if home_team == "COL":
            wind_out_factor = wind_mph * 0.1
        else:
            wind_dir_rad = math.radians(wind_dir)
            wind_out_factor = wind_mph * math.cos(wind_dir_rad) * 0.1

        result = {
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_dir": wind_dir,
            "precip_prob": precip_prob,
            "wind_out_factor": wind_out_factor,
        }
        _weather_cache[cache_key] = {"data": result, "ts": time.time()}
        return result
    except Exception as exc:
        logger.warning("Weather parse failed for %s on %s: %s", home_team, date_str, exc)
        _weather_cache[cache_key] = {"data": None, "ts": time.time()}
        return None


# ---------------------------------------------------------------------------
# Historical weather backfill — one request per team per season
# ---------------------------------------------------------------------------

# CF orientation for each park (degrees that wind blows OUT toward CF)
# 0=N, 90=E, 180=S, 270=W
_CF_ORIENTATION: dict[str, int] = {
    "NYY": 315, "NYM": 45,  "BOS": 270, "TBR": 0,
    "BAL": 315, "TOR": 0,   "CLE": 225, "CHW": 315,
    "DET": 45,  "KCR": 270, "MIN": 0,   "HOU": 0,
    "LAA": 270, "OAK": 270, "SEA": 315, "ATL": 270,
    "MIA": 315, "PHI": 315, "WSN": 315, "CHC": 90,
    "CIN": 270, "MIL": 270, "PIT": 90,  "STL": 315,
    "ARI": 315, "COL": 315, "LAD": 315, "SDP": 270,
    "SFG": 90,  "TEX": 315,
}


def _compute_wind_out_factor(team: str, wind_deg: int) -> float:
    """Compute wind-out factor based on park CF orientation and wind direction.

    Returns float between -1.0 (blowing in) and +1.0 (blowing out).
    """
    cf_dir = _CF_ORIENTATION.get(team, 315)
    diff = abs(wind_deg - cf_dir) % 360
    if diff > 180:
        diff = 360 - diff
    if diff <= 45:
        factor = 1.0 - (diff / 45.0)
        return round(factor, 2)
    elif diff >= 135:
        factor = -1.0 * (1.0 - (180 - diff) / 45.0)
        return round(max(factor, -1.0), 2)
    else:
        return 0.0


def _default_weather() -> dict:
    return {
        "temp_f": 70.0,
        "wind_mph": 8.0,
        "wind_deg": 0,
        "wind_out_factor": 0.0,
        "precip": 0.0,
    }


def fetch_historical_weather_season(
    team: str,
    season: int,
) -> pd.DataFrame:
    """Fetch full season of hourly weather for a team's park.

    Uses Open-Meteo archive API — one request per team per season.
    Caches result to parquet.

    Returns DataFrame with columns:
      date (date), hour (int), temp_f (float),
      wind_mph (float), wind_deg (int), precip (float)
    """
    cache_path = WEATHER_CACHE_DIR / f"{team}_{season}_hourly.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    coords = PARK_COORDS.get(team)
    if not coords:
        return pd.DataFrame()

    lat, lon = coords
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={season}-03-01&end_date={season}-11-30"
        "&hourly=temperature_2m,windspeed_10m,winddirection_10m,precipitation"
        "&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=America/New_York"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Weather fetch failed for %s %d: %s", team, season, exc)
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("windspeed_10m", [])
    wind_dirs = hourly.get("winddirection_10m", [])
    precips = hourly.get("precipitation", [])

    rows = []
    for i, t in enumerate(times):
        try:
            dt = pd.Timestamp(t)
            rows.append({
                "date": dt.date(),
                "hour": dt.hour,
                "temp_f": _safe_float(temps[i] if i < len(temps) else None),
                "wind_mph": _safe_float(winds[i] if i < len(winds) else None),
                "wind_deg": int(wind_dirs[i]) if i < len(wind_dirs) and wind_dirs[i] is not None else 0,
                "precip": _safe_float(precips[i] if i < len(precips) else None),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(cache_path, index=False)
    return df_out


def get_game_weather(
    team: str,
    game_date: str,
    game_hour_local: int = 19,
    season: int = None,
) -> dict:
    """Look up historical weather for a specific game.

    Uses cached hourly data from fetch_historical_weather_season().

    Returns dict with:
      temp_f, wind_mph, wind_deg, wind_out_factor, precip
    """
    if season is None:
        season = pd.Timestamp(game_date).year

    df = fetch_historical_weather_season(team, season)
    if df.empty:
        return _default_weather()

    target_date = pd.Timestamp(game_date).date()
    day_df = df[df["date"] == target_date]
    if day_df.empty:
        return _default_weather()

    hour_df = day_df.copy()
    hour_df = hour_df.assign(hour_diff=(hour_df["hour"] - game_hour_local).abs())
    closest = hour_df.sort_values("hour_diff").iloc[0]

    wind_deg = int(closest.get("wind_deg", 0))
    wind_out = _compute_wind_out_factor(team, wind_deg)

    return {
        "temp_f": float(closest.get("temp_f", 70.0)),
        "wind_mph": float(closest.get("wind_mph", 8.0)),
        "wind_deg": wind_deg,
        "wind_out_factor": wind_out,
        "precip": float(closest.get("precip", 0.0)),
    }


def fetch_all_historical_weather(
    seasons: list[int] = None,
    max_workers: int = 12,
) -> None:
    """Fetch historical weather for all parks and all seasons in parallel.

    Runs once — all results cached to parquet files in WEATHER_CACHE_DIR.

    Usage::

        from line_tracker.model.mlb_data import fetch_all_historical_weather
        fetch_all_historical_weather([2022, 2023, 2024, 2025])
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    teams = list(PARK_COORDS.keys())
    tasks = [(team, season) for team in teams for season in seasons]

    pending = [
        (team, season) for team, season in tasks
        if not (WEATHER_CACHE_DIR / f"{team}_{season}_hourly.parquet").exists()
    ]

    if not pending:
        logger.info("All weather data already cached (%d files)", len(tasks))
        return

    logger.info(
        "Fetching weather for %d team-seasons in parallel (workers=%d)...",
        len(pending), max_workers,
    )

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(fetch_historical_weather_season, team, season): (team, season)
            for team, season in pending
        }
        for future in as_completed(future_to_task):
            team, season = future_to_task[future]
            try:
                df = future.result()
                if df.empty:
                    failed += 1
                    logger.warning("Empty weather data: %s %d", team, season)
                else:
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(
                            "Weather progress: %d/%d completed, %d failed",
                            completed, len(pending), failed,
                        )
            except Exception as exc:
                failed += 1
                logger.warning("Weather fetch error %s %d: %s", team, season, exc)

    logger.info(
        "Weather backfill complete: %d completed, %d failed out of %d total",
        completed, failed, len(pending),
    )


# ---------------------------------------------------------------------------
# First-inning data for NRFI/YRFI model
# ---------------------------------------------------------------------------


def fetch_first_inning_data(
    seasons: list[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch first-inning run data for every completed game in the given seasons.

    For each game, calls the MLB Stats API linescore endpoint to extract
    the runs scored by each team in the first inning.

    Resume-capable: per-season partial caches mean interrupted fetches can
    be continued without re-fetching already-completed game_pks.

    Cache paths
    -----------
    Per-season : CACHE_DIR / "first_inning_{year}.parquet"
    Combined   : CACHE_DIR / "first_inning_{min_s}_{max_s}.parquet"

    Returns
    -------
    DataFrame with columns:
        game_pk, date, home_team, away_team,
        home_sp_id, away_sp_id,
        home_lineup, away_lineup,
        first_inning_home_runs, first_inning_away_runs,
        yrfi (1 = run scored in first inning, 0 = no run),
        season
    """
    min_s, max_s = min(seasons), max(seasons)
    final_cache = CACHE_DIR / f"first_inning_{min_s}_{max_s}.parquet"

    if not force_refresh and final_cache.exists():
        logger.debug("Loading first-inning data from cache: %s", final_cache)
        return pd.read_parquet(final_cache)

    all_season_frames: list[pd.DataFrame] = []

    for year in seasons:
        season_cache = CACHE_DIR / f"first_inning_{year}.parquet"

        # Load game starters (provides game_pk, teams, SPs, lineups)
        print(f"Loading game starters for {year}...")
        try:
            starters = fetch_game_starters(year, force_refresh=force_refresh)
        except Exception as exc:
            logger.error("Failed to load game starters for %d: %s", year, exc)
            continue

        if starters.empty:
            logger.warning("No game starters for season %d; skipping.", year)
            continue

        all_game_pks = set(int(pk) for pk in starters["game_pk"].tolist())
        # Deduplicate before indexing — duplicate game_pks would cause .loc[]
        # to return a DataFrame instead of a Series, corrupting column types.
        starters_lookup = starters.drop_duplicates(subset=["game_pk"]).set_index("game_pk")

        # Load existing partial results for resume
        existing_df: pd.DataFrame | None = None
        already_fetched_pks: set[int] = set()

        if not force_refresh and season_cache.exists():
            try:
                existing_df = pd.read_parquet(season_cache)
                already_fetched_pks = set(int(pk) for pk in existing_df["game_pk"].tolist())
                logger.info(
                    "Resuming %d: %d/%d game_pks already cached.",
                    year, len(already_fetched_pks), len(all_game_pks),
                )
            except Exception as exc:
                logger.warning("Could not load partial cache for %d: %s", year, exc)
                existing_df = None
                already_fetched_pks = set()

        remaining_pks = sorted(all_game_pks - already_fetched_pks)

        if not remaining_pks:
            logger.info("Season %d: all %d games already cached.", year, len(all_game_pks))
            if existing_df is not None:
                all_season_frames.append(existing_df)
            continue

        print(
            f"Season {year}: fetching {len(remaining_pks)} linescore(s) "
            f"({len(already_fetched_pks)} already cached)..."
        )

        new_rows: list[dict] = []
        total = len(remaining_pks)

        def _save_partial() -> None:
            """Merge new_rows with existing cache and persist."""
            parts = []
            if existing_df is not None and not existing_df.empty:
                parts.append(existing_df)
            if new_rows:
                parts.append(pd.DataFrame(new_rows))
            if parts:
                combined = pd.concat(parts, ignore_index=True)
                combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
                combined.to_parquet(season_cache, index=False)

        for i, game_pk in enumerate(remaining_pks, 1):
            if i % 500 == 0:
                print(f"  Fetched {i}/{total} for season {year}...")
                _save_partial()

            try:
                url = f"{MLB_STATS_API}/game/{game_pk}/linescore"
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                innings = data.get("innings", [])
                if not innings:
                    # Postponed / spring training / no play
                    time.sleep(0.05)
                    continue

                first = innings[0]
                home_runs = int(first.get("home", {}).get("runs") or 0)
                away_runs = int(first.get("away", {}).get("runs") or 0)

                try:
                    row_info = starters_lookup.loc[game_pk]
                except KeyError:
                    time.sleep(0.05)
                    continue

                new_rows.append({
                    "game_pk": int(game_pk),
                    "date": pd.to_datetime(row_info["date"]),
                    "home_team": row_info["home_team"],
                    "away_team": row_info["away_team"],
                    "home_sp_id": row_info.get("home_sp_id"),
                    "away_sp_id": row_info.get("away_sp_id"),
                    "home_lineup": row_info.get("home_lineup"),
                    "away_lineup": row_info.get("away_lineup"),
                    "first_inning_home_runs": home_runs,
                    "first_inning_away_runs": away_runs,
                    "yrfi": 1 if (home_runs + away_runs) > 0 else 0,
                    "season": year,
                })

            except Exception as exc:
                logger.debug("Linescore failed for game_pk %s: %s", game_pk, exc)

            time.sleep(0.05)

        # Final save for this season
        _save_partial()

        # Rebuild full season frame from cache (preserves resume consistency)
        if season_cache.exists():
            try:
                season_df = pd.read_parquet(season_cache)
                all_season_frames.append(season_df)
                print(f"Season {year}: {len(season_df)} games with first-inning data.")
            except Exception as exc:
                logger.error("Could not load season cache for %d: %s", year, exc)

    if not all_season_frames:
        return pd.DataFrame()

    result = pd.concat(all_season_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    result.to_parquet(final_cache, index=False)
    logger.info(
        "First-inning data cached: %d games across seasons %s",
        len(result), seasons,
    )
    return result


# ---------------------------------------------------------------------------
# Odds API — player prop lines
# ---------------------------------------------------------------------------

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_PROPS_CACHE_DIR = CACHE_DIR / "props"
_PROPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PROPS_CACHE_TTL_HOURS = 2.0
_ODDS_API_REMAINING_FLOOR = 50  # refuse to fetch below this threshold


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove Jr/Sr/III suffixes for fuzzy matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower().strip()
    for suffix in (" jr", " sr", " iii", " ii", " iv"):
        if ascii_name.endswith(suffix):
            ascii_name = ascii_name[: -len(suffix)].strip()
    return ascii_name


def match_prop_to_prediction(
    odds_name: str,
    candidate_names: list[str],
    threshold: float = 0.85,
) -> str | None:
    """Fuzzy-match an Odds API player name to our internal name list.

    Returns the best match if similarity >= threshold, else None.
    """
    import difflib

    norm_odds = _normalize_name(odds_name)
    best_match: str | None = None
    best_ratio = 0.0
    for candidate in candidate_names:
        ratio = difflib.SequenceMatcher(
            None, norm_odds, _normalize_name(candidate)
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate
    if best_ratio >= threshold:
        return best_match
    return None


def fetch_prop_lines(
    event_ids: list[str],
    markets: list[str],
    api_key: str,
    force_refresh: bool = False,
) -> dict[str, dict]:
    """Fetch player prop lines from The Odds API.

    Costs 1 API call per event per market.
    Markets: batter_home_runs, batter_total_bases, pitcher_strikeouts

    Returns: {event_id: {market: [{player, line, over_price, under_price}]}}
    Cache: props_{date}_{market}.json, 2hr staleness check.
    """
    today = time.strftime("%Y-%m-%d")
    result: dict[str, dict] = {}

    for market in markets:
        cache_path = _PROPS_CACHE_DIR / f"props_{today}_{market}.json"

        cached_data: dict[str, list] = {}
        if not force_refresh and cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_hours < _PROPS_CACHE_TTL_HOURS:
                try:
                    with cache_path.open() as fh:
                        cached_data = json.load(fh)
                    logger.debug("Props cache hit for market %s", market)
                except Exception:
                    cached_data = {}

        fresh_data: dict[str, list] = dict(cached_data)

        for event_id in event_ids:
            if event_id in cached_data:
                for eid, mdata in result.items():
                    pass  # already in cached_data, will merge below
                continue

            url = (
                f"{_ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds"
            )
            params = {
                "apiKey": api_key,
                "regions": "us",
                "markets": market,
                "oddsFormat": "american",
                "bookmakers": "draftkings,fanduel,pinnacle",
            }
            try:
                resp = requests.get(url, params=params, timeout=20)

                remaining = resp.headers.get("x-requests-remaining")
                if remaining is not None:
                    logger.info(
                        "Odds API remaining calls: %s", remaining
                    )
                    try:
                        if int(remaining) < _ODDS_API_REMAINING_FLOOR:
                            logger.warning(
                                "Odds API remaining calls (%s) below floor (%d) — aborting prop fetch",
                                remaining, _ODDS_API_REMAINING_FLOOR,
                            )
                            return result
                    except ValueError:
                        pass

                if resp.status_code == 422:
                    logger.warning("Odds API 422 for event %s market %s", event_id, market)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("Odds API fetch failed for %s/%s: %s", event_id, market, exc)
                continue

            bookmakers = data.get("bookmakers") or []
            player_lines: dict[str, dict] = {}

            for bm in bookmakers:
                for mkt in bm.get("markets") or []:
                    if mkt.get("key") != market:
                        continue
                    for outcome in mkt.get("outcomes") or []:
                        player = outcome.get("description") or outcome.get("name", "")
                        name_lower = outcome.get("name", "").lower()
                        price = int(outcome.get("price", 0))
                        line = float(outcome.get("point") or 0.5)

                        if player not in player_lines:
                            player_lines[player] = {
                                "player": player,
                                "line": line,
                                "over_prices": [],
                                "under_prices": [],
                            }

                        if "over" in name_lower:
                            player_lines[player]["over_prices"].append(price)
                        elif "under" in name_lower:
                            player_lines[player]["under_prices"].append(price)

            summarised: list[dict] = []
            for pdata in player_lines.values():
                over_prices = pdata["over_prices"]
                under_prices = pdata["under_prices"]
                # Best over = lowest juice (closest to 0 from negative side, or lowest positive)
                best_over = min(over_prices, key=lambda p: abs(p)) if over_prices else None
                best_under = min(under_prices, key=lambda p: abs(p)) if under_prices else None
                summarised.append({
                    "player": pdata["player"],
                    "line": pdata["line"],
                    "over_price": best_over,
                    "under_price": best_under,
                })

            fresh_data[event_id] = summarised

        # Persist updated cache
        try:
            with cache_path.open("w") as fh:
                json.dump(fresh_data, fh)
        except Exception as exc:
            logger.warning("Could not write props cache: %s", exc)

        # Merge this market into result
        for event_id in event_ids:
            if event_id not in result:
                result[event_id] = {}
            if event_id in fresh_data:
                result[event_id][market] = fresh_data[event_id]

    return result


def fetch_today_mlb_event_ids(api_key: str) -> list[str]:
    """Fetch today's MLB event IDs from The Odds API (costs 1 API call).

    Returns list of event_id strings for today's games.
    """
    today = time.strftime("%Y-%m-%d")
    cache_path = _PROPS_CACHE_DIR / f"event_ids_{today}.json"

    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < _PROPS_CACHE_TTL_HOURS:
            try:
                with cache_path.open() as fh:
                    return json.load(fh)
            except Exception:
                pass

    url = f"{_ODDS_API_BASE}/sports/baseball_mlb/events"
    params = {"apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=20)
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            logger.info("Odds API remaining calls: %s", remaining)
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        logger.warning("Could not fetch MLB event IDs: %s", exc)
        return []

    event_ids = [e["id"] for e in events if isinstance(e, dict) and "id" in e]

    try:
        with cache_path.open("w") as fh:
            json.dump(event_ids, fh)
    except Exception:
        pass

    return event_ids

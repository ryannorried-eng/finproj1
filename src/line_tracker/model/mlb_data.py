"""
MLB historical game data fetcher using the official MLB Stats API.
Pulls team schedules/results and team game logs from statsapi.mlb.com.
Caches to ~/.cache/line_tracker/mlb/ as parquet.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path.home() / ".cache" / "line_tracker" / "mlb"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"

# Legacy abbreviation lists kept for backward compatibility (may be imported elsewhere)
MLB_TEAMS = [
    "ATL", "ARI", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC",  "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD",  "SEA", "SF",  "STL", "TB",  "TEX", "TOR", "WSH",
]
PYBB_TO_BR = {
    "CWS": "CHW", "KC": "KCR", "SD": "SDP", "SF": "SFG",
    "TB": "TBR", "WSH": "WSN",
}
BR_TO_PYBB = {v: k for k, v in PYBB_TO_BR.items()}

# Map MLB Stats API abbreviations → Baseball Reference abbreviations
_MLB_API_TO_BR: dict[str, str] = {
    "CWS": "CHW",
    "KC":  "KCR",
    "SD":  "SDP",
    "SF":  "SFG",
    "TB":  "TBR",
    "WSH": "WSN",
}


def _to_br(abbrev: str) -> str:
    """Convert an MLB Stats API team abbreviation to Baseball Reference format."""
    return _MLB_API_TO_BR.get(abbrev, abbrev)


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
    DataFrame with columns: game_id, date, season, home_team, away_team,
    home_score, away_score, winner, run_diff, total_runs.
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

                home_abbrev = (
                    home_info.get("team", {}).get("abbreviation", "")
                )
                away_abbrev = (
                    away_info.get("team", {}).get("abbreviation", "")
                )

                date = pd.to_datetime(date_str, errors="coerce")
                if pd.isna(date):
                    continue

                records.append({
                    "date": date,
                    "home_team": _to_br(home_abbrev),
                    "away_team": _to_br(away_abbrev),
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

    df["season"] = season

    # Derived columns
    df["winner"] = df.apply(
        lambda r: "home" if r["home_score"] > r["away_score"] else "away", axis=1
    )
    df["run_diff"] = df["home_score"] - df["away_score"]
    df["total_runs"] = df["home_score"] + df["away_score"]

    df = df[[
        "game_id", "date", "season", "home_team", "away_team",
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
    DataFrame with columns: team, date, home_away, opponent,
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
        br_abbrev = _to_br(abbrev)

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
            date_str = split.get("date", "")
            game_num = split.get("gameNumber", 1)
            stat = split.get("stat", {})
            pit_by_key[(date_str, game_num)] = {
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
                home_away = "H"
            elif is_home is False:
                home_away = "A"
            else:
                home_away = np.nan

            pit = pit_by_key.get((date_str, game_num), {})

            all_rows.append({
                "team": br_abbrev,
                "date": date,
                "home_away": home_away,
                "opponent": _to_br(opponent_abbrev) if opponent_abbrev else np.nan,
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

    # Coerce all numeric columns — keep NaN rather than dropping columns
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
        logger.debug("Loading training data from cache: %s", final_cache)
        return pd.read_parquet(final_cache)

    all_seasons: list[pd.DataFrame] = []

    for year in seasons:
        print(f"Loading MLB data for season {year}...")
        schedule = fetch_schedule_and_results(year, force_refresh=force_refresh)
        logs = fetch_team_game_logs(year, force_refresh=force_refresh)

        if schedule.empty:
            logger.warning("No schedule data for season %d", year)
            continue

        if logs.empty:
            logger.warning("No game logs for season %d", year)
            all_seasons.append(schedule)
            continue

        # Build lookup: (team, date, game_num) → stats
        # For home team stats: look up the game where team=home_team and home_away='H'
        # For away team stats: look up where team=away_team and home_away='A'
        logs["date_str"] = logs["date"].dt.strftime("%Y-%m-%d")

        # Aggregate to handle doubleheaders (average if two games same day)
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

        schedule["date_str"] = schedule["date"].dt.strftime("%Y-%m-%d")

        # Join home team logs
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
            home_logs[["home_team", "date_str", "home_runs_scored", "home_runs_allowed_log",
                        "home_hits", "home_walks", "home_strikeouts", "home_innings_pitched"]],
            on=["home_team", "date_str"],
            how="left",
        )

        # Join away team logs
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
            away_logs[["away_team", "date_str", "away_runs_scored", "away_runs_allowed_log",
                        "away_hits", "away_walks", "away_strikeouts", "away_innings_pitched"]],
            on=["away_team", "date_str"],
            how="left",
        )

        merged["season"] = year
        all_seasons.append(merged)

    if not all_seasons:
        return pd.DataFrame()

    result = pd.concat(all_seasons, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    result.to_parquet(final_cache, index=False)
    logger.info("Cached training data: %d games across seasons %s", len(result), seasons)
    return result

"""
MLB historical game data fetcher using pybaseball.
Pulls team schedules/results and team game logs from Baseball Reference.
Caches to ~/.cache/line_tracker/mlb/ as parquet.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball

pybaseball.cache.enable()

CACHE_DIR = Path.home() / ".cache" / "line_tracker" / "mlb"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

MLB_TEAMS = [
    "ATL", "ARI", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]


def _parse_game_date(date_str: str, season: int) -> pd.Timestamp:
    """Parse a date string from Baseball Reference schedule."""
    s = str(date_str).strip()
    # Remove game number suffix: " (1)" or " (2)" for doubleheaders
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    # Remove day-of-week prefix: "Wednesday, " or "Wed, "
    s = re.sub(r"^\w+,\s*", "", s)
    s = s.strip()
    # Try common formats
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return pd.to_datetime(f"{s} {season}", format=fmt)
        except (ValueError, TypeError):
            pass
    # Fallback: let pandas figure it out
    try:
        return pd.to_datetime(s + f" {season}")
    except Exception:
        return pd.NaT


def _game_number(date_str: str) -> int:
    """Extract doubleheader game number from date string (1 or 2, default 1)."""
    m = re.search(r"\((\d+)\)", str(date_str))
    return int(m.group(1)) if m else 1


def _find_ha_column(df: pd.DataFrame) -> str | None:
    """Find the home/away indicator column (contains '@' for away, '' for home)."""
    for name in ["H/A", "home_away", "homeAway", "HA"]:
        if name in df.columns:
            return name
    # Look for any column that only has '@', '', or NaN values and has '@' present
    for col in df.columns:
        vals = df[col].fillna("").astype(str).str.strip()
        unique = set(vals.unique())
        if unique <= {"@", "", "nan"} and "@" in unique:
            return col
    return None


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
    if cache_path.exists() and not force_refresh:
        logger.debug("Loading schedule from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching schedule for season %d ...", season)
    home_records: list[pd.DataFrame] = []

    for team in MLB_TEAMS:
        try:
            raw = pybaseball.schedule_and_record(season, team)
        except Exception as exc:
            logger.warning("Failed to fetch schedule for %s %d: %s", team, season, exc)
            time.sleep(0.5)
            continue

        if raw is None or raw.empty:
            time.sleep(0.5)
            continue

        # Parse date
        raw = raw.copy()
        raw["_season"] = season
        raw["_team"] = team

        date_col = "Date" if "Date" in raw.columns else raw.columns[1]
        raw["date"] = raw[date_col].apply(lambda d: _parse_game_date(d, season))
        raw["_game_num"] = raw[date_col].apply(_game_number)

        # Find team/opponent columns
        tm_col = next((c for c in ["Tm", "Team"] if c in raw.columns), None)
        opp_col = next((c for c in ["Opp", "Opponent"] if c in raw.columns), None)
        r_col = "R" if "R" in raw.columns else None
        ra_col = "RA" if "RA" in raw.columns else None

        if tm_col is None or opp_col is None:
            logger.warning("Cannot find Tm/Opp columns for %s %d", team, season)
            time.sleep(0.5)
            continue

        # Find home/away column
        ha_col = _find_ha_column(raw)

        if ha_col is not None:
            is_home = raw[ha_col].fillna("").astype(str).str.strip() != "@"
        else:
            # Fallback: check if Tm matches our team (home = team's own venue)
            is_home = pd.Series([True] * len(raw), index=raw.index)
            logger.debug("No H/A column found for %s %d; using all rows as home", team, season)

        home_games = raw[is_home].copy()

        if home_games.empty:
            time.sleep(0.5)
            continue

        # Keep only played games (has scores)
        if r_col and ra_col:
            has_score = (
                pd.to_numeric(home_games[r_col], errors="coerce").notna()
                & pd.to_numeric(home_games[ra_col], errors="coerce").notna()
            )
            home_games = home_games[has_score].copy()
            home_games["_home_score"] = pd.to_numeric(home_games[r_col], errors="coerce")
            home_games["_away_score"] = pd.to_numeric(home_games[ra_col], errors="coerce")
        else:
            logger.warning("Cannot find R/RA columns for %s %d", team, season)
            time.sleep(0.5)
            continue

        home_games["_home_team"] = team
        home_games["_away_team"] = home_games[opp_col].astype(str).str.strip()

        home_records.append(home_games[[
            "date", "_season", "_home_team", "_away_team",
            "_home_score", "_away_score", "_game_num",
        ]])

        time.sleep(0.5)

    if not home_records:
        logger.warning("No schedule data retrieved for season %d", season)
        return pd.DataFrame()

    df = pd.concat(home_records, ignore_index=True)

    # Drop rows with bad dates
    df = df[df["date"].notna()].copy()

    # Drop spring training (before April 1)
    df = df[df["date"] >= pd.Timestamp(f"{season}-04-01")].copy()

    # Build game_id: date_home_away[_gamenum]
    df["game_id"] = (
        df["date"].dt.strftime("%Y-%m-%d")
        + "_" + df["_home_team"]
        + "_" + df["_away_team"]
        + df["_game_num"].apply(lambda n: f"_{n}" if n > 1 else "")
    )

    # Deduplicate (shouldn't happen since we only kept home games)
    df = df.drop_duplicates("game_id").copy()

    # Rename
    df = df.rename(columns={
        "_season": "season",
        "_home_team": "home_team",
        "_away_team": "away_team",
        "_home_score": "home_score",
        "_away_score": "away_score",
    })

    # Compute derived columns
    df["winner"] = df.apply(
        lambda r: "home" if r["home_score"] > r["away_score"] else "away", axis=1
    )
    df["run_diff"] = df["home_score"] - df["away_score"]
    df["total_runs"] = df["home_score"] + df["away_score"]

    df = df[["game_id", "date", "season", "home_team", "away_team",
             "home_score", "away_score", "winner", "run_diff", "total_runs"]].copy()
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
    if cache_path.exists() and not force_refresh:
        logger.debug("Loading team logs from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching team game logs for season %d ...", season)

    batting_records: list[pd.DataFrame] = []
    pitching_records: list[pd.DataFrame] = []

    for team in MLB_TEAMS:
        # --- Batting log ---
        try:
            bat = pybaseball.team_game_logs(season, team, log_type="batting")
        except Exception as exc:
            logger.warning("Batting log failed for %s %d: %s", team, season, exc)
            bat = None
        time.sleep(0.5)

        # --- Pitching log ---
        try:
            pit = pybaseball.team_game_logs(season, team, log_type="pitching")
        except Exception as exc:
            logger.warning("Pitching log failed for %s %d: %s", team, season, exc)
            pit = None
        time.sleep(0.5)

        if bat is not None and not bat.empty:
            bat = bat.copy()
            bat["_team"] = team
            batting_records.append(bat)

        if pit is not None and not pit.empty:
            pit = pit.copy()
            pit["_team"] = team
            pitching_records.append(pit)

    if not batting_records:
        logger.warning("No batting log data for season %d", season)
        return pd.DataFrame()

    # Process batting logs
    bat_all = pd.concat(batting_records, ignore_index=True)
    bat_all = _process_game_log(bat_all, season, "batting")

    # Process pitching logs
    if pitching_records:
        pit_all = pd.concat(pitching_records, ignore_index=True)
        pit_all = _process_game_log(pit_all, season, "pitching")
    else:
        pit_all = pd.DataFrame()

    # Merge batting + pitching on team + date
    if not pit_all.empty:
        merged = bat_all.merge(
            pit_all[["_team", "date", "_game_num", "runs_allowed", "innings_pitched",
                      "hits_allowed", "walks_allowed", "strikeouts_pitched"]],
            on=["_team", "date", "_game_num"],
            how="left",
        )
    else:
        merged = bat_all.copy()
        merged["runs_allowed"] = np.nan
        merged["innings_pitched"] = np.nan
        merged["hits_allowed"] = np.nan
        merged["walks_allowed"] = np.nan
        merged["strikeouts_pitched"] = np.nan

    # Rename team column
    merged = merged.rename(columns={"_team": "team"})

    result = merged[[
        "team", "date", "home_away", "opponent",
        "runs_scored", "runs_allowed", "hits", "walks", "strikeouts",
        "innings_pitched",
    ]].copy()

    result = result.sort_values(["team", "date"]).reset_index(drop=True)
    result.to_parquet(cache_path, index=False)
    logger.info("Cached team logs: %d rows for season %d", len(result), season)
    return result


def _process_game_log(
    df: pd.DataFrame, season: int, log_type: str
) -> pd.DataFrame:
    """Normalize a raw batting or pitching game log DataFrame."""
    df = df.copy()

    # Parse date
    date_col = next((c for c in ["Date", "date"] if c in df.columns), None)
    if date_col:
        df["date"] = df[date_col].apply(lambda d: _parse_game_date(d, season))
        df["_game_num"] = df[date_col].apply(_game_number)
    else:
        df["date"] = pd.NaT
        df["_game_num"] = 1

    df = df[df["date"].notna()].copy()
    df = df[df["date"] >= pd.Timestamp(f"{season}-04-01")].copy()

    # Find home/away column
    ha_col = _find_ha_column(df)
    if ha_col:
        df["home_away"] = df[ha_col].fillna("").astype(str).str.strip().apply(
            lambda v: "A" if v == "@" else "H"
        )
    else:
        df["home_away"] = "H"

    # Find opponent column
    opp_col = next((c for c in ["Opp", "Opponent", "opp"] if c in df.columns), None)
    df["opponent"] = df[opp_col].astype(str).str.strip() if opp_col else ""

    if log_type == "batting":
        # Runs scored by the team
        r_col = _find_numeric_col(df, ["R", "Runs", "RS"])
        df["runs_scored"] = pd.to_numeric(df[r_col], errors="coerce") if r_col else np.nan

        # Hits (batting)
        h_col = _find_numeric_col(df, ["H", "Hits"])
        df["hits"] = pd.to_numeric(df[h_col], errors="coerce") if h_col else np.nan

        # Walks (batting)
        bb_col = _find_numeric_col(df, ["BB", "Walks", "Walk"])
        df["walks"] = pd.to_numeric(df[bb_col], errors="coerce") if bb_col else np.nan

        # Strikeouts (as batters)
        so_col = _find_numeric_col(df, ["SO", "K", "Strikeouts"])
        df["strikeouts"] = pd.to_numeric(df[so_col], errors="coerce") if so_col else np.nan

        return df[["_team", "date", "_game_num", "home_away", "opponent",
                   "runs_scored", "hits", "walks", "strikeouts"]].copy()

    else:  # pitching
        # Runs allowed
        r_col = _find_numeric_col(df, ["R", "RA", "Runs"])
        df["runs_allowed"] = pd.to_numeric(df[r_col], errors="coerce") if r_col else np.nan

        # Innings pitched
        ip_col = _find_numeric_col(df, ["IP", "Inn", "InningsPitched"])
        df["innings_pitched"] = pd.to_numeric(df[ip_col], errors="coerce") if ip_col else np.nan

        # Hits allowed
        h_col = _find_numeric_col(df, ["H", "Hits"])
        df["hits_allowed"] = pd.to_numeric(df[h_col], errors="coerce") if h_col else np.nan

        # Walks allowed
        bb_col = _find_numeric_col(df, ["BB", "Walks"])
        df["walks_allowed"] = pd.to_numeric(df[bb_col], errors="coerce") if bb_col else np.nan

        # Strikeouts by pitchers
        so_col = _find_numeric_col(df, ["SO", "K", "Strikeouts"])
        df["strikeouts_pitched"] = pd.to_numeric(df[so_col], errors="coerce") if so_col else np.nan

        return df[["_team", "date", "_game_num", "home_away", "opponent",
                   "runs_allowed", "innings_pitched",
                   "hits_allowed", "walks_allowed", "strikeouts_pitched"]].copy()


def _find_numeric_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first available column from a list of candidates."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


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
        Re-download pybaseball data even if cached.

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

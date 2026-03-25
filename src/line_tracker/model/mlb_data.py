"""
MLB historical game data fetcher using the official MLB Stats API.
Pulls team schedules/results and team game logs from statsapi.mlb.com.
Caches to ~/.cache/line_tracker/mlb/ as parquet.
"""

from __future__ import annotations

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

        merged["season"] = year
        all_seasons.append(merged)

    if not all_seasons:
        return pd.DataFrame()

    result = pd.concat(all_seasons, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    result.to_parquet(final_cache, index=False)
    logger.info("Cached training data: %d games across seasons %s", len(result), seasons)
    return result


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

    url = (
        f"{MLB_STATS_API}/stats"
        f"?stats=season&group=pitching&gameType=R"
        f"&season={season}&sportId=1&limit=1000"
    )
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch pitcher stats for season %d: %s", season, exc)
        raise

    stat_groups = data.get("stats", [])
    if not stat_groups:
        raise ValueError(f"No pitcher stats returned for season {season}")

    splits = stat_groups[0].get("splits", [])
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

            records.append({
                "pitcher_id": int(pitcher_id),
                "pitcher_name": player.get("fullName", ""),
                "team": _to_canonical(team_info.get("abbreviation", "")),
                "era": _safe_float(stat.get("era")),
                "whip": _safe_float(stat.get("whip")),
                "k9": _safe_float(stat.get("strikeoutsPer9Inn")),
                "bb9": _safe_float(stat.get("walksPer9Inn")),
                "innings": _safe_float(stat.get("inningsPitched")),
                "wins": int(stat.get("wins", 0)),
                "losses": int(stat.get("losses", 0)),
                "games_started": games_started,
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


def fetch_weather(home_team: str, game_date) -> dict | None:
    """Fetch game-time weather via Open-Meteo (no API key required).

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

        return {
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_dir": wind_dir,
            "precip_prob": precip_prob,
            "wind_out_factor": wind_out_factor,
        }
    except Exception as exc:
        logger.warning("Weather parse failed for %s on %s: %s", home_team, date_str, exc)
        return None

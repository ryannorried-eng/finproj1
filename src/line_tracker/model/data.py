"""Fetch NCAAB team-level efficiency ratings from Bart Torvik's database.

Data sources:
  - Team ratings:  barttorvik.com  ``{year}_team_results.json``
  - Four factors:  barttorvik.com  ``teamslicejson.php``
  - Game results:  barttorvik.com  ``getgamestats.php``
  - Schedule:      ESPN public scoreboard API
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from line_tracker.core.logging import get_logger

log = get_logger(__name__)

_BASE = "https://barttorvik.com"
_ESPN_BASE = (
    "https://site.api.espn.com/apis/site/v2/sports"
    "/basketball/mens-college-basketball"
)
_CACHE_DIR = Path.home() / ".line_tracker" / "model_cache"
_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours for current-season data
_HTTP_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> httpx.Client:
    """Return a configured httpx client matching the project convention."""
    return httpx.Client(timeout=_HTTP_TIMEOUT)


def _cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def current_season() -> int:
    """Return the Torvik *season year* for the current date.

    The NCAAB season spans two calendar years.  Torvik labels each season
    by the *spring* year, so Nov 2025 – Apr 2026 is ``2026``.
    """
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year


def _cache_path(label: str, season: int) -> Path:
    return _cache_dir() / f"{season}_{label}.parquet"


def _is_cache_valid(path: Path, season: int) -> bool:
    if not path.exists():
        return False
    if season < current_season():
        return True  # past seasons never expire
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Column maps for raw Torvik JSON arrays
# ---------------------------------------------------------------------------

# ``{year}_team_results.json`` — 45 elements per row
_TEAM_RESULTS_COLS: dict[int, str] = {
    0: "rank",
    1: "team",
    2: "conf",
    3: "record",
    4: "adj_oe",
    6: "adj_de",
    8: "barthag",
    41: "sos",
    44: "adj_t",
}

# ``teamslicejson.php`` — 37 elements per row (four-factors slice)
# Column order: eFG_O, eFG_D, FTR_O, FTR_D, TO_O, TO_D, OR_pct, DR_pct
_SLICE_COLS: dict[int, str] = {
    0: "team",
    7: "efg_o",
    8: "efg_d",
    9: "ftr_o",
    10: "ftr_d",
    11: "to_o",
    12: "to_d",
    13: "or_pct",
}

# ``getgamestats.php`` — 31 elements per row (each game appears twice)
_GAME_COLS: dict[int, str] = {
    0: "date",
    1: "game_type",
    2: "team",
    3: "conf",
    4: "opponent",
    5: "venue",  # H / A / N
    6: "result",
    22: "season",
    24: "game_id",  # composite key like "TeamATeamBMM-DD"
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_team_ratings(season: int | None = None) -> pd.DataFrame:
    """Fetch per-team efficiency ratings for *season*.

    Returns a DataFrame with columns:
        rank, team, conf, record, adj_oe, adj_de, adj_em, adj_t,
        barthag, efg_o, efg_d, to_o, to_d, or_pct, ftr_o, ftr_d, sos,
        wins, losses

    Results are cached to ``~/.line_tracker/model_cache/{season}_ratings.parquet``.
    Current-season caches expire after 6 hours; past-season caches are permanent.
    """
    if season is None:
        season = current_season()

    cache = _cache_path("ratings", season)
    if _is_cache_valid(cache, season):
        log.info("Loading cached ratings for %d", season)
        return pd.read_parquet(cache)

    log.info("Fetching team ratings for %d from barttorvik", season)

    with _get_client() as client:
        # --- core ratings ---------------------------------------------------
        resp = client.get(f"{_BASE}/{season}_team_results.json")
        resp.raise_for_status()
        raw_ratings = resp.json()

        rows = []
        for row in raw_ratings:
            entry = {}
            for idx, col in _TEAM_RESULTS_COLS.items():
                entry[col] = row[idx] if idx < len(row) else None
            rows.append(entry)
        df_ratings = pd.DataFrame(rows)

        # --- four factors ----------------------------------------------------
        resp2 = client.get(
            f"{_BASE}/teamslicejson.php",
            params={"year": season, "json": 1, "type": "R"},
        )
        resp2.raise_for_status()
        raw_slice = resp2.json()

        slice_rows = []
        for row in raw_slice:
            entry = {}
            for idx, col in _SLICE_COLS.items():
                entry[col] = row[idx] if idx < len(row) else None
            slice_rows.append(entry)
        df_slice = pd.DataFrame(slice_rows)

    # --- merge ---------------------------------------------------------------
    df = df_ratings.merge(df_slice, on="team", how="left")

    # derived columns
    df["adj_em"] = df["adj_oe"] - df["adj_de"]
    _parse_record(df)

    # coerce numeric columns
    numeric = [
        "rank", "adj_oe", "adj_de", "adj_em", "adj_t", "barthag",
        "efg_o", "efg_d", "ftr_o", "ftr_d", "to_o", "to_d", "or_pct",
        "sos", "wins", "losses",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    col_order = [
        "rank", "team", "conf", "record", "adj_oe", "adj_de", "adj_em",
        "adj_t", "barthag", "efg_o", "efg_d", "to_o", "to_d", "or_pct",
        "ftr_o", "ftr_d", "sos", "wins", "losses",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    df.to_parquet(cache, index=False)
    log.info("Cached %d team ratings for %d", len(df), season)
    return df


def fetch_game_results(season: int) -> pd.DataFrame:
    """Fetch game-by-game results for *season*.

    Returns a DataFrame with columns:
        date, home_team, away_team, home_score, away_score, margin,
        neutral_site, game_id
    """
    cache = _cache_path("games", season)
    if _is_cache_valid(cache, season):
        log.info("Loading cached game results for %d", season)
        return pd.read_parquet(cache)

    log.info("Fetching game results for %d from barttorvik", season)

    with _get_client() as client:
        resp = client.get(
            f"{_BASE}/getgamestats.php",
            params={"year": season},
        )
        resp.raise_for_status()
        raw = resp.json()

    # Build one row per game (deduplicate the two-perspectives-per-game).
    # Keep only the "home" perspective; for neutral-site games, keep the row
    # where venue == "H" or, if both are "N", keep the first alphabetically.
    games: dict[int, dict] = {}
    for row in raw:
        entry: dict = {}
        for idx, col in _GAME_COLS.items():
            entry[col] = row[idx] if idx < len(row) else None

        gid = entry["game_id"]
        venue = entry["venue"]
        neutral = venue == "N"

        # Parse result string like "W, 96-62" or "L, 62-96"
        score_match = re.match(r"([WL]),\s*(\d+)-(\d+)", entry.get("result") or "")
        if not score_match:
            continue
        outcome = score_match.group(1)
        s1, s2 = int(score_match.group(2)), int(score_match.group(3))
        team_score = s1 if outcome == "W" else s2
        opp_score = s2 if outcome == "W" else s1

        if gid in games:
            # Already have the other perspective — skip unless this is home
            if venue == "H":
                games[gid] = _build_game_row(
                    entry, team_score, opp_score, neutral,
                )
            continue

        if venue == "H" or venue == "N":
            games[gid] = _build_game_row(
                entry, team_score, opp_score, neutral,
            )
        else:
            # This row is the away team; flip perspective
            games[gid] = {
                "date": entry["date"],
                "home_team": entry["opponent"],
                "away_team": entry["team"],
                "home_score": opp_score,
                "away_score": team_score,
                "margin": opp_score - team_score,
                "neutral_site": neutral,
                "game_id": gid,
            }

    df = pd.DataFrame(list(games.values()))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df.to_parquet(cache, index=False)
    log.info("Cached %d game results for %d", len(df), season)
    return df


def fetch_schedule() -> pd.DataFrame:
    """Fetch upcoming NCAAB games (today + next 7 days).

    Returns a DataFrame with columns:
        date, home_team, away_team, neutral_site, espn_id

    Used to match model predictions against Odds API events.
    """
    today = datetime.now(timezone.utc)
    dates = [(today.date().isoformat().replace("-", ""))]
    for i in range(1, 8):
        d = today.date().toordinal() + i
        dates.append(
            datetime.fromordinal(d).strftime("%Y%m%d")
        )

    all_games: list[dict] = []
    with _get_client() as client:
        for date_str in dates:
            try:
                resp = client.get(
                    f"{_ESPN_BASE}/scoreboard",
                    params={"dates": date_str, "limit": 200},
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError:
                log.warning("Failed to fetch ESPN schedule for %s", date_str)
                continue

            for event in data.get("events", []):
                competition = event.get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])
                if len(competitors) < 2:
                    continue
                home = away = None
                for c in competitors:
                    info = {
                        "team": c.get("team", {}).get("displayName", ""),
                    }
                    if c.get("homeAway") == "home":
                        home = info
                    else:
                        away = info
                if not home or not away:
                    continue
                all_games.append({
                    "date": event.get("date", ""),
                    "home_team": home["team"],
                    "away_team": away["team"],
                    "neutral_site": competition.get("neutralSite", False),
                    "espn_id": event.get("id", ""),
                })

    df = pd.DataFrame(all_games)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    log.info("Found %d upcoming NCAAB games", len(df))
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_game_row(
    entry: dict, team_score: int, opp_score: int, neutral: bool,
) -> dict:
    return {
        "date": entry["date"],
        "home_team": entry["team"],
        "away_team": entry["opponent"],
        "home_score": team_score,
        "away_score": opp_score,
        "margin": team_score - opp_score,
        "neutral_site": neutral,
        "game_id": entry["game_id"],
    }


def _parse_record(df: pd.DataFrame) -> None:
    """Extract *wins* and *losses* from the ``record`` column in-place."""
    parts = df["record"].str.split("-", n=1, expand=True)
    if parts.shape[1] >= 2:
        df["wins"] = pd.to_numeric(parts[0], errors="coerce")
        df["losses"] = pd.to_numeric(parts[1], errors="coerce")
    else:
        df["wins"] = None
        df["losses"] = None

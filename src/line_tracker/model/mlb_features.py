"""
MLB feature engineering — v2 pitcher-aware + bullpen + weather + home/away splits.
All rolling windows are computed from prior games only (no future leakage).
v1 features are preserved; v2 adds ~17 new columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hardcoded 2025 park factors (index 100 = neutral)
PARK_FACTORS: dict[str, dict[str, int]] = {
    "COL": {"runs": 115, "hr": 123}, "BOS": {"runs": 107, "hr": 103},
    "CIN": {"runs": 106, "hr": 111}, "TEX": {"runs": 105, "hr": 108},
    "MIL": {"runs": 104, "hr": 102}, "CHC": {"runs": 103, "hr": 105},
    "BAL": {"runs": 103, "hr": 106}, "ATL": {"runs": 102, "hr": 101},
    "NYY": {"runs": 102, "hr": 110}, "PIT": {"runs": 101, "hr": 98},
    "PHI": {"runs": 101, "hr": 104}, "HOU": {"runs": 100, "hr": 99},
    "LAD": {"runs": 100, "hr": 101}, "SFG": {"runs": 99, "hr": 93},
    "STL": {"runs": 99, "hr": 97},  "MIN": {"runs": 99, "hr": 102},
    "ARI": {"runs": 99, "hr": 100}, "DET": {"runs": 98, "hr": 96},
    "TBR": {"runs": 98, "hr": 95},  "NYM": {"runs": 98, "hr": 97},
    "SDP": {"runs": 97, "hr": 95},  "TOR": {"runs": 97, "hr": 96},
    "SEA": {"runs": 97, "hr": 93},  "CLE": {"runs": 97, "hr": 94},
    "LAA": {"runs": 96, "hr": 96},  "WSN": {"runs": 96, "hr": 98},
    "CHW": {"runs": 96, "hr": 97},  "MIA": {"runs": 95, "hr": 91},
    "KCR": {"runs": 95, "hr": 94},  "OAK": {"runs": 94, "hr": 90},
}

# Feature columns produced by build_feature_matrix
# v1 columns (indices 0-20) are unchanged; v2 appends indices 21-37.
FEATURE_COLUMNS = [
    # v1 — team form (21 features)
    "home_runs_scored_r15",
    "home_runs_allowed_r15",
    "home_run_diff_r15",
    "home_k_rate_r15",
    "home_bb_rate_r15",
    "home_run_diff_r10",
    "away_runs_scored_r15",
    "away_runs_allowed_r15",
    "away_run_diff_r15",
    "away_k_rate_r15",
    "away_bb_rate_r15",
    "away_run_diff_r10",
    "offense_diff",
    "defense_diff",
    "form_diff",
    "is_home",
    "home_days_rest",
    "away_days_rest",
    "rest_advantage",
    "park_factor_runs",
    "park_factor_hr",
    # v2 — starting pitcher rotation quality (7 features)
    "home_rotation_era",
    "away_rotation_era",
    "home_rotation_k9",
    "away_rotation_k9",
    "home_rotation_whip",
    "away_rotation_whip",
    "sp_era_diff",
    # v2 — bullpen ERA proxy (3 features)
    "home_bullpen_era_r7",
    "away_bullpen_era_r7",
    "bullpen_era_diff",
    # v2 — home/away splits (4 features)
    "home_team_runs_scored_home_r15",
    "away_team_runs_scored_away_r15",
    "home_team_runs_allowed_home_r15",
    "away_team_runs_allowed_away_r15",
    # v2 — weather placeholders (3 features; overwritten at prediction time)
    "wind_out_factor",
    "temp_f",
    "precip_prob",
]


def compute_team_rolling_stats(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling form statistics per team with no future leakage.

    Parameters
    ----------
    game_logs:
        Per-team per-game rows with columns: team, date, runs_scored,
        runs_allowed, hits, walks, strikeouts.

    Returns
    -------
    Same DataFrame with added rolling stat columns:
    runs_scored_r15, runs_allowed_r15, run_diff_r15,
    k_rate_r15, bb_rate_r15, run_diff_r10.
    """
    result_parts = []

    for _team, group in game_logs.groupby("team"):
        group = group.sort_values("date").copy()

        # Apply pd.to_numeric on plain Series before shift — guarantees float output
        s_scored = pd.to_numeric(group["runs_scored"], errors="coerce").shift(1)
        s_allowed = pd.to_numeric(group["runs_allowed"], errors="coerce").shift(1)
        s_hits = pd.to_numeric(group["hits"], errors="coerce").shift(1)
        s_walks = pd.to_numeric(group["walks"], errors="coerce").shift(1)
        s_strikeouts = pd.to_numeric(group["strikeouts"], errors="coerce").shift(1)

        run_diff = s_scored - s_allowed

        group["runs_scored_r15"] = s_scored.rolling(15, min_periods=15).mean()
        group["runs_allowed_r15"] = s_allowed.rolling(15, min_periods=15).mean()
        group["run_diff_r15"] = run_diff.rolling(15, min_periods=15).mean()

        # K% and BB% proxies using H+BB+SO as denominator
        denom = s_hits + s_walks + s_strikeouts
        safe_denom = denom.replace(0, np.nan)
        group["k_rate_r15"] = (s_strikeouts / safe_denom).rolling(15, min_periods=15).mean()
        group["bb_rate_r15"] = (s_walks / safe_denom).rolling(15, min_periods=15).mean()

        group["run_diff_r10"] = run_diff.rolling(10, min_periods=10).mean()

        result_parts.append(group)

    return pd.concat(result_parts, ignore_index=True)


_ROT_DEFAULTS = {"rotation_era": 4.20, "rotation_k9": 8.5, "rotation_whip": 1.30}
_BP_DEFAULT_ERA = 4.20


def _compute_rotation_quality(seasons: list) -> dict:
    """Compute weighted avg ERA/K9/WHIP for top-5 starters per (team, season).

    Returns dict: {(team_str, season_int): {rotation_era, rotation_k9, rotation_whip}}
    Falls back to league averages on any error.
    """
    from line_tracker.model.mlb_data import fetch_pitcher_season_stats

    result: dict = {}
    for season in seasons:
        try:
            pitcher_df = fetch_pitcher_season_stats(int(season))
        except Exception as exc:
            logger.warning("Could not load pitcher stats for %d: %s", season, exc)
            continue

        for team, group in pitcher_df.groupby("team"):
            top5 = group.nlargest(5, "games_started")
            total_gs = top5["games_started"].sum()
            if total_gs == 0:
                result[(str(team), int(season))] = _ROT_DEFAULTS.copy()
                continue
            weights = top5["games_started"] / total_gs
            result[(str(team), int(season))] = {
                "rotation_era": float((top5["era"].fillna(4.20) * weights).sum()),
                "rotation_k9": float((top5["k9"].fillna(8.5) * weights).sum()),
                "rotation_whip": float((top5["whip"].fillna(1.30) * weights).sum()),
            }
    return result


def _compute_bullpen_index(seasons: list) -> dict:
    """Build (team_str, pd.Timestamp) → {era_r7, era_r15} lookup.

    Falls back to league-average ERA on missing data.
    """
    from line_tracker.model.mlb_data import fetch_bullpen_stats

    result: dict = {}
    for season in seasons:
        try:
            bp_df = fetch_bullpen_stats(int(season))
        except Exception as exc:
            logger.warning("Could not load bullpen stats for %d: %s", season, exc)
            continue

        bp_df["era_r7"] = pd.to_numeric(bp_df["era_r7"], errors="coerce").fillna(_BP_DEFAULT_ERA)
        bp_df["era_r15"] = pd.to_numeric(bp_df["era_r15"], errors="coerce").fillna(_BP_DEFAULT_ERA)

        for row in bp_df.itertuples(index=False):
            key = (str(row.team), row.date)
            result[key] = {
                "era_r7": float(row.era_r7),
                "era_r15": float(row.era_r15),
            }
    return result


def _compute_home_away_splits(game_logs: pd.DataFrame) -> dict:
    """Build (team_str, pd.Timestamp) → home/away rolling split stats lookup.

    Uses the correct approach: compute rolling on the home-only (or away-only)
    game subset, then forward-fill to all dates.

    Returns dict: {(team, date): {runs_scored_home_r15, runs_allowed_home_r15,
                                   runs_scored_away_r15, runs_allowed_away_r15}}
    """
    result_parts: list[pd.DataFrame] = []

    for team, group in game_logs.groupby("team"):
        group = group.sort_values("date").copy()

        # --- Home split ---
        home_only = group[group["home_away"] == "H"].copy().sort_values("date")
        if len(home_only) > 0:
            sh_scored = home_only["runs_scored"].shift(1)
            sh_allowed = home_only["runs_allowed"].shift(1)
            home_only["rs_home_r15"] = sh_scored.rolling(15, min_periods=15).mean()
            home_only["ra_home_r15"] = sh_allowed.rolling(15, min_periods=15).mean()
        else:
            home_only["rs_home_r15"] = np.nan
            home_only["ra_home_r15"] = np.nan

        # --- Away split ---
        away_only = group[group["home_away"] == "A"].copy().sort_values("date")
        if len(away_only) > 0:
            sa_scored = away_only["runs_scored"].shift(1)
            sa_allowed = away_only["runs_allowed"].shift(1)
            away_only["rs_away_r15"] = sa_scored.rolling(15, min_periods=15).mean()
            away_only["ra_away_r15"] = sa_allowed.rolling(15, min_periods=15).mean()
        else:
            away_only["rs_away_r15"] = np.nan
            away_only["ra_away_r15"] = np.nan

        # Merge back to full date list and forward-fill
        temp = group[["date"]].copy()
        if len(home_only) > 0:
            temp = temp.merge(
                home_only[["date", "rs_home_r15", "ra_home_r15"]],
                on="date", how="left",
            )
        else:
            temp["rs_home_r15"] = np.nan
            temp["ra_home_r15"] = np.nan

        if len(away_only) > 0:
            temp = temp.merge(
                away_only[["date", "rs_away_r15", "ra_away_r15"]],
                on="date", how="left",
            )
        else:
            temp["rs_away_r15"] = np.nan
            temp["ra_away_r15"] = np.nan

        temp = temp.sort_values("date")
        for col in ["rs_home_r15", "ra_home_r15", "rs_away_r15", "ra_away_r15"]:
            temp[col] = temp[col].ffill()
        temp["team"] = team
        result_parts.append(
            temp[["team", "date", "rs_home_r15", "ra_home_r15", "rs_away_r15", "ra_away_r15"]]
        )

    if not result_parts:
        return {}

    splits_df = pd.concat(result_parts, ignore_index=True)
    splits_df = (
        splits_df.sort_values(["team", "date"])
        .drop_duplicates(subset=["team", "date"], keep="first")
    )

    result: dict = {}
    for row in splits_df.itertuples(index=False):
        key = (str(row.team), row.date)
        result[key] = {
            "runs_scored_home_r15": float(row.rs_home_r15) if not pd.isna(row.rs_home_r15) else None,
            "runs_allowed_home_r15": float(row.ra_home_r15) if not pd.isna(row.ra_home_r15) else None,
            "runs_scored_away_r15": float(row.rs_away_r15) if not pd.isna(row.rs_away_r15) else None,
            "runs_allowed_away_r15": float(row.ra_away_r15) if not pd.isna(row.ra_away_r15) else None,
        }
    return result


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build training feature matrix and targets from raw game-level data.

    Parameters
    ----------
    df:
        Raw training data from load_mlb_training_data().

    Returns
    -------
    (X, y) where X has feature columns and y has target columns
    (home_win, run_diff, total_runs).
    """
    df = df.copy()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure season column exists (needed for rotation/bullpen lookups)
    if "season" not in df.columns:
        df["season"] = df["date"].dt.year

    unique_seasons = sorted(df["season"].dropna().unique().tolist())

    # ---------------------------------------------------------------------------
    # Step 1: Build per-team per-game view for rolling stats (v1)
    # ---------------------------------------------------------------------------
    # Use schedule scores for runs (most reliable); fill from game logs if present
    home_view = pd.DataFrame({
        "team": df["home_team"],
        "date": df["date"],
        "home_away": "H",
        "runs_scored": df["home_score"],
        "runs_allowed": df["away_score"],
        "hits": df.get("home_hits", pd.Series(np.nan, index=df.index)),
        "walks": df.get("home_walks", pd.Series(np.nan, index=df.index)),
        "strikeouts": df.get("home_strikeouts", pd.Series(np.nan, index=df.index)),
    })

    away_view = pd.DataFrame({
        "team": df["away_team"],
        "date": df["date"],
        "home_away": "A",
        "runs_scored": df["away_score"],
        "runs_allowed": df["home_score"],
        "hits": df.get("away_hits", pd.Series(np.nan, index=df.index)),
        "walks": df.get("away_walks", pd.Series(np.nan, index=df.index)),
        "strikeouts": df.get("away_strikeouts", pd.Series(np.nan, index=df.index)),
    })

    game_logs = pd.concat([home_view, away_view], ignore_index=True)
    game_logs = game_logs.sort_values(["team", "date"]).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # Step 2: Compute rolling stats (v1)
    # ---------------------------------------------------------------------------
    rolling = compute_team_rolling_stats(game_logs)

    # Build a lookup: (team, date) → rolling stats
    # Deduplicate so .loc always returns a Series (scalar per column), not a DataFrame.
    # For doubleheaders with the same (team, date), keep first entry — rolling stats
    # computed before any games that day, which is appropriate for all games on that date.
    rolling = rolling.sort_values(["team", "date"]).drop_duplicates(
        subset=["team", "date"], keep="first"
    )
    rolling_index = rolling.set_index(["team", "date"])

    stat_cols = [
        "runs_scored_r15", "runs_allowed_r15", "run_diff_r15",
        "k_rate_r15", "bb_rate_r15", "run_diff_r10",
    ]

    # ---------------------------------------------------------------------------
    # Step 3: Compute days_rest per team (v1)
    # ---------------------------------------------------------------------------
    all_dates = pd.concat([
        pd.DataFrame({"team": df["home_team"], "date": df["date"]}),
        pd.DataFrame({"team": df["away_team"], "date": df["date"]}),
    ]).drop_duplicates().sort_values(["team", "date"])

    all_dates["days_rest"] = (
        all_dates.groupby("team")["date"]
        .diff()
        .dt.days
        .fillna(1)
        .clip(1, 7)
    )
    rest_index = all_dates.set_index(["team", "date"])["days_rest"]

    # ---------------------------------------------------------------------------
    # Step 4: v2 — precompute rotation quality, bullpen ERA, home/away splits
    # ---------------------------------------------------------------------------
    rotation_lookup = _compute_rotation_quality(unique_seasons)
    bullpen_index = _compute_bullpen_index(unique_seasons)
    splits_index = _compute_home_away_splits(game_logs)

    # ---------------------------------------------------------------------------
    # Step 5: Assemble matchup feature rows
    # ---------------------------------------------------------------------------
    feature_rows = []
    target_rows = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        season = int(row["season"])

        try:
            home_stats = rolling_index.loc[(home, date), stat_cols]
            away_stats = rolling_index.loc[(away, date), stat_cols]
        except KeyError:
            continue  # Skip if rolling stats not available

        # Home rolling stats (v1)
        home_rs15 = home_stats["runs_scored_r15"]
        home_ra15 = home_stats["runs_allowed_r15"]
        home_rd15 = home_stats["run_diff_r15"]
        home_k15 = home_stats["k_rate_r15"]
        home_bb15 = home_stats["bb_rate_r15"]
        home_rd10 = home_stats["run_diff_r10"]

        # Away rolling stats (v1)
        away_rs15 = away_stats["runs_scored_r15"]
        away_ra15 = away_stats["runs_allowed_r15"]
        away_rd15 = away_stats["run_diff_r15"]
        away_k15 = away_stats["k_rate_r15"]
        away_bb15 = away_stats["bb_rate_r15"]
        away_rd10 = away_stats["run_diff_r10"]

        # Days rest (v1)
        try:
            home_rest = float(rest_index.loc[(home, date)])
        except KeyError:
            home_rest = 1.0
        try:
            away_rest = float(rest_index.loc[(away, date)])
        except KeyError:
            away_rest = 1.0

        # Park factors (v1, normalized)
        pf = PARK_FACTORS.get(home, {"runs": 100, "hr": 100})
        park_runs = pf["runs"] / 100.0
        park_hr = pf["hr"] / 100.0

        # v2 — rotation quality
        home_rot = rotation_lookup.get((home, season), _ROT_DEFAULTS)
        away_rot = rotation_lookup.get((away, season), _ROT_DEFAULTS)
        home_rotation_era = home_rot["rotation_era"]
        away_rotation_era = away_rot["rotation_era"]
        sp_era_diff = away_rotation_era - home_rotation_era

        # v2 — bullpen ERA
        home_bp = bullpen_index.get((home, date), {})
        away_bp = bullpen_index.get((away, date), {})
        home_bullpen_era_r7 = home_bp.get("era_r7", _BP_DEFAULT_ERA)
        away_bullpen_era_r7 = away_bp.get("era_r7", _BP_DEFAULT_ERA)
        bullpen_era_diff = away_bullpen_era_r7 - home_bullpen_era_r7

        # v2 — home/away splits
        home_splits = splits_index.get((home, date), {})
        away_splits = splits_index.get((away, date), {})
        home_rs_home_r15 = home_splits.get("runs_scored_home_r15") or home_rs15
        home_ra_home_r15 = home_splits.get("runs_allowed_home_r15") or home_ra15
        away_rs_away_r15 = away_splits.get("runs_scored_away_r15") or away_rs15
        away_ra_away_r15 = away_splits.get("runs_allowed_away_r15") or away_ra15

        feat = {
            # v1 features
            "home_runs_scored_r15": home_rs15,
            "home_runs_allowed_r15": home_ra15,
            "home_run_diff_r15": home_rd15,
            "home_k_rate_r15": home_k15,
            "home_bb_rate_r15": home_bb15,
            "home_run_diff_r10": home_rd10,
            "away_runs_scored_r15": away_rs15,
            "away_runs_allowed_r15": away_ra15,
            "away_run_diff_r15": away_rd15,
            "away_k_rate_r15": away_k15,
            "away_bb_rate_r15": away_bb15,
            "away_run_diff_r10": away_rd10,
            "offense_diff": home_rs15 - away_rs15,
            "defense_diff": home_ra15 - away_ra15,
            "form_diff": home_rd10 - away_rd10,
            "is_home": 1,
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
            "rest_advantage": home_rest - away_rest,
            "park_factor_runs": park_runs,
            "park_factor_hr": park_hr,
            # v2 — rotation
            "home_rotation_era": home_rotation_era,
            "away_rotation_era": away_rotation_era,
            "home_rotation_k9": home_rot["rotation_k9"],
            "away_rotation_k9": away_rot["rotation_k9"],
            "home_rotation_whip": home_rot["rotation_whip"],
            "away_rotation_whip": away_rot["rotation_whip"],
            "sp_era_diff": sp_era_diff,
            # v2 — bullpen
            "home_bullpen_era_r7": home_bullpen_era_r7,
            "away_bullpen_era_r7": away_bullpen_era_r7,
            "bullpen_era_diff": bullpen_era_diff,
            # v2 — home/away splits
            "home_team_runs_scored_home_r15": home_rs_home_r15,
            "away_team_runs_scored_away_r15": away_rs_away_r15,
            "home_team_runs_allowed_home_r15": home_ra_home_r15,
            "away_team_runs_allowed_away_r15": away_ra_away_r15,
            # v2 — weather placeholders (neutral values for training)
            "wind_out_factor": 0.0,
            "temp_f": 72.0,
            "precip_prob": 0.0,
        }
        feature_rows.append(feat)

        tgt = {
            "home_win": 1 if row["home_score"] > row["away_score"] else 0,
            "run_diff": float(row["home_score"] - row["away_score"]),
            "total_runs": float(row["home_score"] + row["away_score"]),
        }
        target_rows.append(tgt)

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    y = pd.DataFrame(target_rows)

    # Drop rows where any v1 feature is NaN (v2 features have fallbacks so rarely NaN)
    valid = X.notna().all(axis=1)
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    logger.info(
        "Feature matrix: %d rows, %d features (from %d games, dropped %d NaN rows)",
        len(X), X.shape[1], len(df), (~valid).sum(),
    )
    return X, y

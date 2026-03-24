"""
MLB feature engineering for v1 team-form model.
All rolling windows are computed from prior games only (no future leakage).
No pitcher-level features in v1.
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
FEATURE_COLUMNS = [
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

    # ---------------------------------------------------------------------------
    # Step 1: Build per-team per-game view for rolling stats
    # ---------------------------------------------------------------------------
    # Use schedule scores for runs (most reliable); fill from game logs if present
    home_view = pd.DataFrame({
        "team": df["home_team"],
        "date": df["date"],
        "runs_scored": df["home_score"],
        "runs_allowed": df["away_score"],
        "hits": df.get("home_hits", pd.Series(np.nan, index=df.index)),
        "walks": df.get("home_walks", pd.Series(np.nan, index=df.index)),
        "strikeouts": df.get("home_strikeouts", pd.Series(np.nan, index=df.index)),
    })

    away_view = pd.DataFrame({
        "team": df["away_team"],
        "date": df["date"],
        "runs_scored": df["away_score"],
        "runs_allowed": df["home_score"],
        "hits": df.get("away_hits", pd.Series(np.nan, index=df.index)),
        "walks": df.get("away_walks", pd.Series(np.nan, index=df.index)),
        "strikeouts": df.get("away_strikeouts", pd.Series(np.nan, index=df.index)),
    })

    game_logs = pd.concat([home_view, away_view], ignore_index=True)
    game_logs = game_logs.sort_values(["team", "date"]).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # Step 2: Compute rolling stats
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
    # Step 3: Compute days_rest per team
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
    # Step 4: Assemble matchup feature rows
    # ---------------------------------------------------------------------------
    feature_rows = []
    target_rows = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        try:
            home_stats = rolling_index.loc[(home, date), stat_cols]
            away_stats = rolling_index.loc[(away, date), stat_cols]
        except KeyError:
            continue  # Skip if rolling stats not available

        # Home rolling stats
        home_rs15 = home_stats["runs_scored_r15"]
        home_ra15 = home_stats["runs_allowed_r15"]
        home_rd15 = home_stats["run_diff_r15"]
        home_k15 = home_stats["k_rate_r15"]
        home_bb15 = home_stats["bb_rate_r15"]
        home_rd10 = home_stats["run_diff_r10"]

        # Away rolling stats
        away_rs15 = away_stats["runs_scored_r15"]
        away_ra15 = away_stats["runs_allowed_r15"]
        away_rd15 = away_stats["run_diff_r15"]
        away_k15 = away_stats["k_rate_r15"]
        away_bb15 = away_stats["bb_rate_r15"]
        away_rd10 = away_stats["run_diff_r10"]

        # Days rest
        try:
            home_rest = float(rest_index.loc[(home, date)])
        except KeyError:
            home_rest = 1.0
        try:
            away_rest = float(rest_index.loc[(away, date)])
        except KeyError:
            away_rest = 1.0

        # Park factors (normalized)
        pf = PARK_FACTORS.get(home, {"runs": 100, "hr": 100})
        park_runs = pf["runs"] / 100.0
        park_hr = pf["hr"] / 100.0

        feat = {
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

    # Drop rows where any feature is NaN
    valid = X.notna().all(axis=1)
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    logger.info(
        "Feature matrix: %d rows (from %d games, dropped %d NaN rows)",
        len(X), len(df), (~valid).sum(),
    )
    return X, y

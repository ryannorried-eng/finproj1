"""
NRFI/YRFI feature engineering.

Builds per-game features from first-inning data, pitcher stats,
batter stats, and park/weather context.

All rolling windows are computed using only prior games (shift(1) before
rolling) to prevent data leakage.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from line_tracker.model.mlb_features import PARK_FACTORS

logger = logging.getLogger(__name__)

LEAGUE_AVG_YRFI_RATE = 0.565   # ~56.5% historical NRFI/YRFI average
LEAGUE_AVG_ERA       = 4.20
LEAGUE_AVG_WHIP      = 1.30
LEAGUE_AVG_K9        = 8.50
LEAGUE_AVG_BB9       = 3.20
LEAGUE_AVG_OBP       = 0.320

NRFI_FEATURE_COLUMNS = [
    "home_sp_fi_era",
    "away_sp_fi_era",
    "home_sp_k9",
    "away_sp_k9",
    "home_sp_bb9",
    "away_sp_bb9",
    "home_sp_whip",
    "away_sp_whip",
    "home_sp_fi_yrfi_rate_r10",
    "away_sp_fi_yrfi_rate_r10",
    "home_top3_obp_r15",
    "away_top3_obp_r15",
    "home_top3_k_rate_r15",
    "away_top3_k_rate_r15",
    "home_team_obp_r15",
    "away_team_obp_r15",
    "park_factor_runs",
    "temp_f",
    "wind_out_factor",
    "is_day_game",
    "home_team_yrfi_rate_r15",
    "away_team_yrfi_rate_r15",
]


def _parse_lineup_top3_ids(lineup_json: str | None) -> list[int]:
    """Return player IDs for batting positions 1-3 from a lineup JSON string."""
    if not lineup_json:
        return []
    try:
        players = json.loads(lineup_json)
        top3 = [p["id"] for p in players if p.get("batting_order", 99) <= 3]
        return top3[:3]
    except Exception:
        return []


def _safe_float(val, default: float = np.nan) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _sp_rolling_stats(first_inning_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling first-inning stats for each starting pitcher.

    For each game, builds a record for the home SP (runs allowed =
    first_inning_away_runs) and the away SP (runs allowed =
    first_inning_home_runs).  Then computes rolling 10-start windows
    with shift(1) to prevent leakage.

    Returns
    -------
    DataFrame indexed by game_pk with columns:
        home_sp_fi_era_r10, home_sp_fi_yrfi_rate_r10,
        away_sp_fi_era_r10, away_sp_fi_yrfi_rate_r10
    """
    df = first_inning_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Build flat table of SP appearances
    home_starts = df[["game_pk", "date", "home_sp_id", "first_inning_away_runs", "yrfi"]].copy()
    home_starts = home_starts.rename(columns={
        "home_sp_id": "pitcher_id",
        "first_inning_away_runs": "runs_allowed",
    })
    home_starts["side"] = "home"

    away_starts = df[["game_pk", "date", "away_sp_id", "first_inning_home_runs", "yrfi"]].copy()
    away_starts = away_starts.rename(columns={
        "away_sp_id": "pitcher_id",
        "first_inning_home_runs": "runs_allowed",
    })
    away_starts["side"] = "away"

    sp_df = pd.concat([home_starts, away_starts], ignore_index=True)
    sp_df = sp_df.dropna(subset=["pitcher_id"])
    sp_df["pitcher_id"] = sp_df["pitcher_id"].astype(int)
    sp_df = sp_df.sort_values(["pitcher_id", "date"]).reset_index(drop=True)

    # Rolling 10-start stats (shift(1) = exclude current game)
    sp_df["sp_fi_era_r10"] = (
        sp_df.groupby("pitcher_id")["runs_allowed"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean() * 9)
    )
    sp_df["sp_fi_yrfi_rate_r10"] = (
        sp_df.groupby("pitcher_id")["yrfi"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Split home/away back and join to game_pk
    home_rolling = sp_df[sp_df["side"] == "home"][
        ["game_pk", "sp_fi_era_r10", "sp_fi_yrfi_rate_r10"]
    ].rename(columns={
        "sp_fi_era_r10": "home_sp_fi_era",
        "sp_fi_yrfi_rate_r10": "home_sp_fi_yrfi_rate_r10",
    })

    away_rolling = sp_df[sp_df["side"] == "away"][
        ["game_pk", "sp_fi_era_r10", "sp_fi_yrfi_rate_r10"]
    ].rename(columns={
        "sp_fi_era_r10": "away_sp_fi_era",
        "sp_fi_yrfi_rate_r10": "away_sp_fi_yrfi_rate_r10",
    })

    return home_rolling.merge(away_rolling, on="game_pk", how="outer")


def _team_yrfi_rolling(first_inning_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 15-game YRFI rates for each team (home and away sides).

    Returns DataFrame with columns: game_pk, home_team_yrfi_rate_r15,
    away_team_yrfi_rate_r15.
    """
    df = first_inning_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Home-team YRFI rolling
    home_df = df[["game_pk", "date", "home_team", "yrfi"]].copy()
    home_df = home_df.sort_values(["home_team", "date"])
    home_df["home_team_yrfi_rate_r15"] = (
        home_df.groupby("home_team")["yrfi"]
        .transform(lambda x: x.shift(1).rolling(15, min_periods=1).mean())
    )

    # Away-team YRFI rolling
    away_df = df[["game_pk", "date", "away_team", "yrfi"]].copy()
    away_df = away_df.sort_values(["away_team", "date"])
    away_df["away_team_yrfi_rate_r15"] = (
        away_df.groupby("away_team")["yrfi"]
        .transform(lambda x: x.shift(1).rolling(15, min_periods=1).mean())
    )

    merged = home_df[["game_pk", "home_team_yrfi_rate_r15"]].merge(
        away_df[["game_pk", "away_team_yrfi_rate_r15"]],
        on="game_pk",
        how="outer",
    )
    return merged


def _batter_obp_rolling(
    first_inning_df: pd.DataFrame,
    batter_logs: pd.DataFrame | None,
    batter_stats: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute rolling top-3 batter OBP and K-rate features.

    Prefers game-level logs for rolling computation.  Falls back to season
    batter stats when logs are unavailable.

    Returns DataFrame with columns:
        game_pk, home_top3_obp_r15, away_top3_obp_r15,
        home_top3_k_rate_r15, away_top3_k_rate_r15,
        home_team_obp_r15, away_team_obp_r15
    """
    fi = first_inning_df.copy()
    fi["date"] = pd.to_datetime(fi["date"])

    # Pre-compute OBP per batter per game from logs
    if batter_logs is not None and not batter_logs.empty:
        logs = batter_logs.copy()
        logs["date"] = pd.to_datetime(logs["date"])
        logs["batter_id"] = logs["batter_id"].astype(int)

        denom = (logs["at_bats"].fillna(0) + logs["walks"].fillna(0))
        logs["game_obp"] = np.where(
            denom > 0,
            (logs["hits"].fillna(0) + logs["walks"].fillna(0)) / denom,
            np.nan,
        )
        # k_rate approximation: (AB - H) / PA (non-contact per PA)
        logs["game_k_rate"] = np.where(
            logs["plate_appearances"].fillna(0) > 0,
            (logs["at_bats"].fillna(0) - logs["hits"].fillna(0))
            / logs["plate_appearances"].fillna(0),
            np.nan,
        )

        logs = logs.sort_values(["batter_id", "date"])
        logs["rolling_obp_r15"] = (
            logs.groupby("batter_id")["game_obp"]
            .transform(lambda x: x.shift(1).rolling(15, min_periods=1).mean())
        )
        logs["rolling_k_rate_r15"] = (
            logs.groupby("batter_id")["game_k_rate"]
            .transform(lambda x: x.shift(1).rolling(15, min_periods=1).mean())
        )

        # Build lookup: (batter_id, game_pk) → rolling stats
        batter_lookup = logs.set_index(["batter_id", "game_pk"])[
            ["rolling_obp_r15", "rolling_k_rate_r15"]
        ]
    else:
        batter_lookup = None

    # Season-level fallback OBP (dict batter_id → obp)
    season_obp: dict[int, float] = {}
    if batter_stats is not None and not batter_stats.empty:
        for bid, row in batter_stats.iterrows():
            obp = _safe_float(row.get("obp"))
            if not np.isnan(obp):
                season_obp[int(bid)] = obp

    rows = []
    for _, game in fi.iterrows():
        game_pk = int(game["game_pk"])

        result: dict[str, float] = {
            "game_pk": game_pk,
            "home_top3_obp_r15": np.nan,
            "away_top3_obp_r15": np.nan,
            "home_top3_k_rate_r15": np.nan,
            "away_top3_k_rate_r15": np.nan,
            "home_team_obp_r15": np.nan,
            "away_team_obp_r15": np.nan,
        }

        for side in ("home", "away"):
            batter_ids = _parse_lineup_top3_ids(game.get(f"{side}_lineup"))

            obps: list[float] = []
            k_rates: list[float] = []
            for bid in batter_ids:
                obp = np.nan
                k_rate = np.nan
                # Try rolling game-level lookup first
                if batter_lookup is not None:
                    try:
                        row_data = batter_lookup.loc[(bid, game_pk)]
                        obp = _safe_float(row_data["rolling_obp_r15"])
                        k_rate = _safe_float(row_data["rolling_k_rate_r15"])
                    except KeyError:
                        pass
                # Fall back to season stat
                if np.isnan(obp):
                    obp = season_obp.get(bid, np.nan)
                if not np.isnan(obp):
                    obps.append(obp)
                if not np.isnan(k_rate):
                    k_rates.append(k_rate)

            if obps:
                result[f"{side}_top3_obp_r15"] = float(np.mean(obps))
            if k_rates:
                result[f"{side}_top3_k_rate_r15"] = float(np.mean(k_rates))
            # team_obp_r15 uses top3 as proxy (full lineup stats unavailable)
            result[f"{side}_team_obp_r15"] = result[f"{side}_top3_obp_r15"]

        rows.append(result)

    return pd.DataFrame(rows)


def build_nrfi_features(
    first_inning_df: pd.DataFrame,
    pitcher_stats: dict[int, pd.DataFrame] | None = None,
    batter_stats: dict[int, pd.DataFrame] | None = None,
    batter_logs: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build NRFI/YRFI feature matrix for model training or prediction.

    Parameters
    ----------
    first_inning_df:
        Output of fetch_first_inning_data().  Must contain game_pk, date,
        home_team, away_team, home_sp_id, away_sp_id, home_lineup,
        away_lineup, first_inning_home_runs, first_inning_away_runs,
        yrfi, season.
    pitcher_stats:
        Dict mapping season year → DataFrame from fetch_pitcher_season_stats().
        Used for SP ERA, WHIP, K9, BB9.  None = use league averages.
    batter_stats:
        Dict mapping season year → DataFrame from fetch_batter_season_stats().
        Used for fallback OBP when game logs are unavailable.  None = use
        league average.
    batter_logs:
        Dict mapping season year → DataFrame from fetch_batter_game_logs().
        Used for rolling per-game OBP.  None = fall back to batter_stats.

    Returns
    -------
    DataFrame with NRFI_FEATURE_COLUMNS + 'yrfi' target.
    """
    if first_inning_df is None or first_inning_df.empty:
        raise ValueError("first_inning_df is empty")

    fi = first_inning_df.copy()
    fi["date"] = pd.to_datetime(fi["date"])
    fi = fi.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. SP rolling first-inning stats (from first_inning_df itself)
    # ------------------------------------------------------------------
    sp_rolling = _sp_rolling_stats(fi)
    fi = fi.merge(sp_rolling, on="game_pk", how="left")

    # ------------------------------------------------------------------
    # 2. SP season stats (ERA, WHIP, K9, BB9) from pitcher_stats dicts
    # ------------------------------------------------------------------
    # Build pitcher lookup: pitcher_id → stats (prefer the season the game was in)
    pitcher_lookup: dict[int, dict] = {}
    if pitcher_stats:
        for year, ps_df in pitcher_stats.items():
            if ps_df is None or ps_df.empty:
                continue
            for pid, row in ps_df.iterrows():
                # Prefer stats from the same season; later entries overwrite earlier
                pitcher_lookup[int(pid)] = {
                    "era":  _safe_float(row.get("era"),  LEAGUE_AVG_ERA),
                    "whip": _safe_float(row.get("whip"), LEAGUE_AVG_WHIP),
                    "k9":   _safe_float(row.get("k9"),   LEAGUE_AVG_K9),
                    "bb9":  _safe_float(row.get("bb9"),  LEAGUE_AVG_BB9),
                }

    def _sp_stat(pitcher_id, stat_key: str, default: float) -> float:
        if pd.isna(pitcher_id):
            return default
        stats = pitcher_lookup.get(int(pitcher_id), {})
        val = _safe_float(stats.get(stat_key), default)
        return val if not np.isnan(val) else default

    fi["home_sp_k9"]   = fi["home_sp_id"].apply(lambda x: _sp_stat(x, "k9",   LEAGUE_AVG_K9))
    fi["away_sp_k9"]   = fi["away_sp_id"].apply(lambda x: _sp_stat(x, "k9",   LEAGUE_AVG_K9))
    fi["home_sp_bb9"]  = fi["home_sp_id"].apply(lambda x: _sp_stat(x, "bb9",  LEAGUE_AVG_BB9))
    fi["away_sp_bb9"]  = fi["away_sp_id"].apply(lambda x: _sp_stat(x, "bb9",  LEAGUE_AVG_BB9))
    fi["home_sp_whip"] = fi["home_sp_id"].apply(lambda x: _sp_stat(x, "whip", LEAGUE_AVG_WHIP))
    fi["away_sp_whip"] = fi["away_sp_id"].apply(lambda x: _sp_stat(x, "whip", LEAGUE_AVG_WHIP))

    # Fill SP fi_era from rolling (fall back to season ERA / 9)
    for side in ("home", "away"):
        col = f"{side}_sp_fi_era"
        era_col = f"{side}_sp_fi_era"
        season_fallback = fi[f"{side}_sp_id"].apply(
            lambda x: _sp_stat(x, "era", LEAGUE_AVG_ERA) / 9.0
        )
        if era_col in fi.columns:
            fi[era_col] = fi[era_col].fillna(season_fallback * 9)
        else:
            fi[era_col] = season_fallback * 9

    # Fill fi_yrfi_rate from rolling (fall back to league average)
    for col in ("home_sp_fi_yrfi_rate_r10", "away_sp_fi_yrfi_rate_r10"):
        if col in fi.columns:
            fi[col] = fi[col].fillna(LEAGUE_AVG_YRFI_RATE)
        else:
            fi[col] = LEAGUE_AVG_YRFI_RATE

    # ------------------------------------------------------------------
    # 3. Team rolling YRFI rates
    # ------------------------------------------------------------------
    team_yrfi = _team_yrfi_rolling(fi)
    fi = fi.merge(team_yrfi, on="game_pk", how="left")
    fi["home_team_yrfi_rate_r15"] = fi["home_team_yrfi_rate_r15"].fillna(LEAGUE_AVG_YRFI_RATE)
    fi["away_team_yrfi_rate_r15"] = fi["away_team_yrfi_rate_r15"].fillna(LEAGUE_AVG_YRFI_RATE)

    # ------------------------------------------------------------------
    # 4. Batter OBP / K-rate features
    # ------------------------------------------------------------------
    # Flatten batter_stats and batter_logs across seasons
    flat_batter_stats: pd.DataFrame | None = None
    if batter_stats:
        parts = [df for df in batter_stats.values() if df is not None and not df.empty]
        if parts:
            combined = pd.concat(parts)
            # Keep most recent record per batter_id
            flat_batter_stats = combined.groupby(combined.index).last()

    flat_batter_logs: pd.DataFrame | None = None
    if batter_logs:
        parts = [df for df in batter_logs.values() if df is not None and not df.empty]
        if parts:
            flat_batter_logs = pd.concat(parts, ignore_index=True)

    batter_features = _batter_obp_rolling(fi, flat_batter_logs, flat_batter_stats)
    fi = fi.merge(batter_features, on="game_pk", how="left")

    # Fill batter features with league averages
    for col, default in [
        ("home_top3_obp_r15",    LEAGUE_AVG_OBP),
        ("away_top3_obp_r15",    LEAGUE_AVG_OBP),
        ("home_team_obp_r15",    LEAGUE_AVG_OBP),
        ("away_team_obp_r15",    LEAGUE_AVG_OBP),
        ("home_top3_k_rate_r15", 0.22),
        ("away_top3_k_rate_r15", 0.22),
    ]:
        if col not in fi.columns:
            fi[col] = default
        else:
            fi[col] = fi[col].fillna(default)

    # ------------------------------------------------------------------
    # 5. Park factors
    # ------------------------------------------------------------------
    fi["park_factor_runs"] = fi["home_team"].map(
        lambda t: PARK_FACTORS.get(str(t), {}).get("runs", 100) / 100.0
    )

    # ------------------------------------------------------------------
    # 6. Weather features (use defaults if not in df)
    # ------------------------------------------------------------------
    if "temp_f" not in fi.columns:
        fi["temp_f"] = 72.0
    else:
        fi["temp_f"] = fi["temp_f"].fillna(72.0)

    if "wind_out_factor" not in fi.columns:
        fi["wind_out_factor"] = 0.0
    else:
        fi["wind_out_factor"] = fi["wind_out_factor"].fillna(0.0)

    # ------------------------------------------------------------------
    # 7. Is day game (default False since first_inning_df has no time)
    # ------------------------------------------------------------------
    fi["is_day_game"] = 0

    # ------------------------------------------------------------------
    # Select final feature columns + target
    # ------------------------------------------------------------------
    output_cols = NRFI_FEATURE_COLUMNS + ["yrfi", "game_pk", "date"]
    missing = [c for c in NRFI_FEATURE_COLUMNS if c not in fi.columns]
    if missing:
        logger.warning("Missing feature columns (will fill with defaults): %s", missing)
        for c in missing:
            fi[c] = 0.0

    return fi[output_cols].copy()

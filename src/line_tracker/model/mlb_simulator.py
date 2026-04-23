"""
Monte Carlo plate appearance simulator.
Simulates N plate appearances between a batter and pitcher
using Statcast outcome probabilities.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "line_tracker" / "mlb"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_STATCAST_CACHE_TTL_HOURS = 24.0

# 2024-2025 MLB league-average PA outcome rates
LEAGUE_AVG_OUTCOMES: dict[str, float] = {
    "K":      0.226,
    "BB":     0.085,
    "HR":     0.031,
    "single": 0.149,
    "double": 0.044,
    "triple": 0.005,
    "out":    0.460,
}

# Total bases per outcome
_TB_MAP: dict[str, int] = {
    "K": 0, "BB": 0, "out": 0,
    "single": 1, "double": 2, "triple": 3, "HR": 4,
}

_HIT_OUTCOMES = {"single", "double", "triple", "HR"}

# Statcast event → PA outcome bucket
_EVENT_MAP: dict[str, str] = {
    "strikeout":                    "K",
    "strikeout_double_play":        "K",
    "walk":                         "BB",
    "hit_by_pitch":                 "BB",
    "home_run":                     "HR",
    "single":                       "single",
    "double":                       "double",
    "triple":                       "triple",
    "field_out":                    "out",
    "grounded_into_double_play":    "out",
    "double_play":                  "out",
    "triple_play":                  "out",
    "force_out":                    "out",
    "fielders_choice":              "out",
    "fielders_choice_out":          "out",
    "sac_fly":                      "out",
    "sac_fly_double_play":          "out",
    "sac_bunt":                     "out",
    "sac_bunt_double_play":         "out",
    "field_error":                  "out",
    "other_out":                    "out",
    "catcher_interf":               "BB",
}

_TERMINAL_EVENTS = set(_EVENT_MAP.keys())


def _load_parquet_cache(path: Path) -> Any | None:
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except Exception:
        return None


def _parse_batter_statcast(df: Any) -> dict:
    """Convert a raw Statcast batter DataFrame into an outcome profile dict."""
    import pandas as pd

    if df is None or (hasattr(df, "empty") and df.empty):
        return {}

    terminal = df[df["events"].notna() & df["events"].isin(_TERMINAL_EVENTS)].copy()
    if len(terminal) == 0:
        return {}

    pa_count = len(terminal)
    counts: dict[str, int] = {k: 0 for k in LEAGUE_AVG_OUTCOMES}
    for ev in terminal["events"]:
        bucket = _EVENT_MAP.get(str(ev), "out")
        counts[bucket] = counts.get(bucket, 0) + 1

    outcomes = {k: v / pa_count for k, v in counts.items()}

    # Handedness splits
    vs_rhp = vs_lhp = {}
    if "p_throws" in terminal.columns:
        rhp = terminal[terminal["p_throws"] == "R"]
        lhp = terminal[terminal["p_throws"] == "L"]
        vs_rhp = {"hr_rate": len(rhp[rhp["events"] == "home_run"]) / max(len(rhp), 1)}
        vs_lhp = {"hr_rate": len(lhp[lhp["events"] == "home_run"]) / max(len(lhp), 1)}

    # Contact quality
    batted = terminal[terminal["events"].isin({"single", "double", "triple", "HR", "field_out",
                                               "grounded_into_double_play", "double_play",
                                               "force_out", "fielders_choice", "fielders_choice_out",
                                               "sac_fly", "field_error", "other_out"})]
    avg_launch_speed = avg_launch_angle = barrel_rate = None
    if "launch_speed" in df.columns and len(batted) > 0:
        ls = df.loc[batted.index, "launch_speed"].dropna()
        la = df.loc[batted.index, "launch_angle"].dropna() if "launch_angle" in df.columns else pd.Series(dtype=float)
        avg_launch_speed = float(ls.mean()) if len(ls) > 0 else None
        avg_launch_angle = float(la.mean()) if len(la) > 0 else None

    if "barrel" in df.columns and len(batted) > 0:
        barrels = df.loc[batted.index, "barrel"].fillna(0)
        barrel_rate = float(barrels.mean())

    return {
        "pa_count": pa_count,
        "pa_outcomes": outcomes,
        "vs_rhp": vs_rhp,
        "vs_lhp": vs_lhp,
        "avg_launch_speed": avg_launch_speed,
        "avg_launch_angle": avg_launch_angle,
        "barrel_rate": barrel_rate,
        "k_rate_statcast": outcomes.get("K", 0.0),
        "bb_rate_statcast": outcomes.get("BB", 0.0),
        "hr_rate_statcast": outcomes.get("HR", 0.0),
    }


def _parse_pitcher_statcast(df: Any) -> dict:
    """Convert a raw Statcast pitcher DataFrame into an outcome profile dict."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return {}

    terminal = df[df["events"].notna() & df["events"].isin(_TERMINAL_EVENTS)].copy()
    if len(terminal) == 0:
        return {}

    pa_count = len(terminal)
    counts: dict[str, int] = {k: 0 for k in LEAGUE_AVG_OUTCOMES}
    for ev in terminal["events"]:
        bucket = _EVENT_MAP.get(str(ev), "out")
        counts[bucket] = counts.get(bucket, 0) + 1

    outcomes = {k: v / pa_count for k, v in counts.items()}

    # Pitch mix
    pitch_mix: dict[str, float] = {}
    if "pitch_type" in df.columns:
        pt_counts = df["pitch_type"].dropna().value_counts(normalize=True)
        pitch_mix = {str(k): round(float(v), 4) for k, v in pt_counts.items()}

    avg_release_speed = None
    if "release_speed" in df.columns:
        rs = df["release_speed"].dropna()
        if len(rs) > 0:
            avg_release_speed = float(rs.mean())

    # Ground ball / fly ball
    gb_events = {"field_out", "grounded_into_double_play", "double_play", "force_out",
                 "fielders_choice", "fielders_choice_out"}
    fb_events = {"home_run", "sac_fly", "sac_fly_double_play"}
    gb_count = sum(1 for ev in terminal["events"] if str(ev) in gb_events)
    fb_count = sum(1 for ev in terminal["events"] if str(ev) in fb_events)
    gb_rate = gb_count / pa_count
    fb_rate = fb_count / pa_count

    return {
        "pa_count": pa_count,
        "pa_outcomes": outcomes,
        "pitch_mix": pitch_mix,
        "avg_release_speed": avg_release_speed,
        "ground_ball_rate": round(gb_rate, 4),
        "fly_ball_rate": round(fb_rate, 4),
        "k_rate": outcomes.get("K", 0.0),
        "bb_rate": outcomes.get("BB", 0.0),
        "hr_rate": outcomes.get("HR", 0.0),
    }


def fetch_statcast_batter_profile(
    batter_id: int,
    season: int = 2026,
    force_refresh: bool = False,
) -> dict:
    """Fetch and cache Statcast profile for a batter.

    Uses pybaseball.statcast_batter() for the season.
    Returns outcome probability distributions.
    Cache: statcast_batter_{id}_{season}.parquet, 24hr staleness.
    """
    cache_path = CACHE_DIR / f"statcast_batter_{batter_id}_{season}.parquet"

    if not force_refresh and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < _STATCAST_CACHE_TTL_HOURS:
            df = _load_parquet_cache(cache_path)
            if df is not None:
                return _parse_batter_statcast(df)

    # Determine season date range
    start_date = f"{season}-03-01"
    end_date = time.strftime("%Y-%m-%d") if season == 2026 else f"{season}-11-01"

    try:
        import pybaseball as pb
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pb.statcast_batter(start_date, end_date, player_id=batter_id)
    except Exception as exc:
        logger.warning("statcast_batter fetch failed for %d: %s", batter_id, exc)
        return {}

    if df is None or (hasattr(df, "empty") and df.empty) or (hasattr(df, "columns") and list(df.columns) == ["Host not in allowlist"]):
        logger.debug("No Statcast data for batter %d season %d", batter_id, season)
        return {}

    try:
        df.to_parquet(cache_path, index=False)
    except Exception as exc:
        logger.debug("Could not cache Statcast batter data: %s", exc)

    return _parse_batter_statcast(df)


def fetch_statcast_pitcher_profile(
    pitcher_id: int,
    season: int = 2026,
    force_refresh: bool = False,
) -> dict:
    """Fetch and cache Statcast profile for a pitcher.

    Returns outcome rates and pitch mix.
    Cache: statcast_pitcher_{id}_{season}.parquet, 24hr staleness.
    """
    cache_path = CACHE_DIR / f"statcast_pitcher_{pitcher_id}_{season}.parquet"

    if not force_refresh and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < _STATCAST_CACHE_TTL_HOURS:
            df = _load_parquet_cache(cache_path)
            if df is not None:
                return _parse_pitcher_statcast(df)

    start_date = f"{season}-03-01"
    end_date = time.strftime("%Y-%m-%d") if season == 2026 else f"{season}-11-01"

    try:
        import pybaseball as pb
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pb.statcast_pitcher(start_date, end_date, player_id=pitcher_id)
    except Exception as exc:
        logger.warning("statcast_pitcher fetch failed for %d: %s", pitcher_id, exc)
        return {}

    if df is None or (hasattr(df, "empty") and df.empty) or (hasattr(df, "columns") and list(df.columns) == ["Host not in allowlist"]):
        logger.debug("No Statcast data for pitcher %d season %d", pitcher_id, season)
        return {}

    try:
        df.to_parquet(cache_path, index=False)
    except Exception as exc:
        logger.debug("Could not cache Statcast pitcher data: %s", exc)

    return _parse_pitcher_statcast(df)


def simulate_pa(
    batter_profile: dict,
    pitcher_profile: dict,
    n_simulations: int = 1000,
    handedness_adjust: bool = True,
) -> dict:
    """Simulate N plate appearances between batter and pitcher.

    Uses blended outcome probabilities:
    - batter_weight = 0.45
    - pitcher_weight = 0.45
    - league_average_weight = 0.10

    Returns aggregated results over n_simulations.
    """
    batter_outcomes = batter_profile.get("pa_outcomes") or {}
    pitcher_outcomes = pitcher_profile.get("pa_outcomes") or {}

    batter_w = 0.45
    pitcher_w = 0.45
    league_w = 0.10

    blended: dict[str, float] = {}
    for outcome in LEAGUE_AVG_OUTCOMES:
        b_rate = batter_outcomes.get(outcome, LEAGUE_AVG_OUTCOMES[outcome])
        p_rate = pitcher_outcomes.get(outcome, LEAGUE_AVG_OUTCOMES[outcome])
        lg_rate = LEAGUE_AVG_OUTCOMES[outcome]
        blended[outcome] = b_rate * batter_w + p_rate * pitcher_w + lg_rate * league_w

    total = sum(blended.values())
    if total <= 0:
        normalized = dict(LEAGUE_AVG_OUTCOMES)
    else:
        normalized = {k: v / total for k, v in blended.items()}

    keys = list(normalized.keys())
    probs = np.array([normalized[k] for k in keys], dtype=float)
    probs = probs / probs.sum()  # ensure exact sum=1 for numpy

    outcome_draws = np.random.choice(keys, size=n_simulations, p=probs)

    k_count = int(np.sum(outcome_draws == "K"))
    bb_count = int(np.sum(outcome_draws == "BB"))
    hr_count = int(np.sum(outcome_draws == "HR"))
    hit_count = int(sum(np.sum(outcome_draws == h) for h in _HIT_OUTCOMES))
    tb_total = int(sum(_TB_MAP.get(o, 0) for o in outcome_draws))

    return {
        "k_rate":            k_count / n_simulations,
        "bb_rate":           bb_count / n_simulations,
        "hr_rate":           hr_count / n_simulations,
        "hit_rate":          hit_count / n_simulations,
        "avg_bases_per_pa":  tb_total / n_simulations,
        "hr_prob_per_pa":    hr_count / n_simulations,
        "expected_tb_per_pa": tb_total / n_simulations,
        "sim_count":         n_simulations,
        "blended_outcomes":  {k: round(v, 5) for k, v in normalized.items()},
    }


def simulate_game_props(
    batter_id: int,
    pitcher_id: int,
    expected_pas: float,
    n_simulations: int = 10000,
    season: int = 2026,
) -> dict:
    """Simulate a batter's full game against a pitcher.

    Runs simulate_pa() then scales to expected_pas.
    Returns prop probabilities.
    """
    batter_profile = fetch_statcast_batter_profile(batter_id, season=season)
    pitcher_profile = fetch_statcast_pitcher_profile(pitcher_id, season=season)

    pa_result = simulate_pa(
        batter_profile=batter_profile,
        pitcher_profile=pitcher_profile,
        n_simulations=n_simulations,
    )

    # Per-PA rates from simulation
    hr_prob_per_pa = pa_result["hr_rate"]
    hit_prob_per_pa = pa_result["hit_rate"]
    k_prob_per_pa = pa_result["k_rate"]
    tb_per_pa = pa_result["expected_tb_per_pa"]

    # Scale to full game using expected_pas
    # We simulate n_simulations full games, each with Poisson(expected_pas) PAs
    rng = np.random.default_rng()
    pas_per_game = rng.poisson(lam=expected_pas, size=n_simulations)

    blended = pa_result["blended_outcomes"]
    keys = list(blended.keys())
    probs = np.array([blended[k] for k in keys], dtype=float)
    probs = probs / probs.sum()

    # For each game, sample outcomes for each PA
    hr_games = 0
    tb_totals: list[float] = []
    hit_totals: list[float] = []
    k_totals: list[float] = []

    for n_pa in pas_per_game:
        if n_pa == 0:
            tb_totals.append(0)
            hit_totals.append(0)
            k_totals.append(0)
            continue
        game_outcomes = np.random.choice(keys, size=int(n_pa), p=probs)
        hrs = int(np.sum(game_outcomes == "HR"))
        hits = int(sum(np.sum(game_outcomes == h) for h in _HIT_OUTCOMES))
        ks = int(np.sum(game_outcomes == "K"))
        tb = int(sum(_TB_MAP.get(o, 0) for o in game_outcomes))
        if hrs >= 1:
            hr_games += 1
        tb_totals.append(tb)
        hit_totals.append(hits)
        k_totals.append(ks)

    tb_arr = np.array(tb_totals)
    hit_arr = np.array(hit_totals)
    k_arr = np.array(k_totals)

    return {
        "hr_prob":        round(hr_games / n_simulations, 4),
        "over_1_5_tb":    round(float(np.mean(tb_arr >= 2)), 4),
        "over_2_5_tb":    round(float(np.mean(tb_arr >= 3)), 4),
        "over_0_5_hits":  round(float(np.mean(hit_arr >= 1)), 4),
        "expected_tb":    round(float(np.mean(tb_arr)), 3),
        "expected_hits":  round(float(np.mean(hit_arr)), 3),
        "k_prob":         round(float(np.mean(k_arr >= 1)), 4),
        "sim_count":      n_simulations,
        "used_statcast_batter":  batter_profile.get("pa_count", 0) > 0,
        "used_statcast_pitcher": pitcher_profile.get("pa_count", 0) > 0,
    }

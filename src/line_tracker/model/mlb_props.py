"""
MLB player prop predictions for HR, total bases, and pitcher Ks.
Uses only free MLB Stats API data — no Odds API calls.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# League-average constants (2024 MLB)
_LEAGUE_AVG_HR_PER9: float = 1.1
_LEAGUE_AVG_HITS_PER9: float = 8.5

# Expected PAs by batting order slot (slots 1-9)
_PA_BY_ORDER: dict[int, float] = {
    1: 4.5, 2: 4.3, 3: 4.2, 4: 4.0, 5: 3.9,
    6: 3.8, 7: 3.7, 8: 3.6, 9: 3.5,
}
_DEFAULT_PA = 3.8


def _pitcher_row(pitcher_id: int | None, pitcher_stats: pd.DataFrame | None) -> pd.Series | None:
    if pitcher_id is None or pitcher_stats is None or pitcher_stats.empty:
        return None
    if pitcher_id in pitcher_stats.index:
        return pitcher_stats.loc[pitcher_id]
    return None


def _parse_lineup(lineup_json: str | list | None) -> list[dict]:
    """Return list of {id, name, batting_order} from lineup JSON or list."""
    if lineup_json is None:
        return []
    if isinstance(lineup_json, list):
        raw = lineup_json
    else:
        try:
            raw = json.loads(lineup_json)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    result = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            result.append({
                "id":           int(item.get("id") or item.get("batter_id") or 0),
                "name":         str(item.get("name") or item.get("batter_name") or ""),
                "batting_order": int(item.get("batting_order") or item.get("order") or (i + 1)),
            })
        elif isinstance(item, (int, float)):
            result.append({"id": int(item), "name": "", "batting_order": i + 1})
    return result


def predict_hr_props(
    game_lineup: str | list | None,
    batter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame | None,
    weather: dict | None,
    park_factor: float = 1.0,
) -> list[dict]:
    """Predict HR props for each batter in a lineup.

    Parameters
    ----------
    game_lineup:
        JSON string or list of batter dicts (id, name, batting_order).
    batter_stats:
        DataFrame indexed by batter_id with hr_rate, hr_per_air, ab_per_hr columns.
    pitcher_stats:
        DataFrame indexed by pitcher_id for the opposing pitcher.
    weather:
        Dict with wind_out_factor, temp_f keys.
    park_factor:
        Park HR factor from PARK_FACTORS (e.g. 1.10 for COL).

    Returns
    -------
    List of prop dicts ranked by hr_prob descending.
    """
    lineup = _parse_lineup(game_lineup)
    if not lineup or batter_stats.empty:
        return []

    wind_out = float((weather or {}).get("wind_out_factor", 0.0))
    temp_f   = float((weather or {}).get("temp_f", 72.0))

    # Pitcher modifier: their hr_per9 vs league avg
    pitcher_mod = 1.0
    if pitcher_stats is not None and not pitcher_stats.empty:
        # We receive pitcher_stats as the full DF; the opposing pitcher must be
        # looked up externally, so here we accept a pre-sliced single-row Series
        # OR the full df (in which case we can't know who's pitching without id).
        # For the full-DF case, default to 1.0 — callers should pass a single row.
        pass

    wind_mod = 1.0 + (wind_out * 0.15)
    temp_mod = 1.0 + ((temp_f - 72.0) * 0.002)

    results = []
    for batter in lineup:
        bid = batter["id"]
        if bid == 0 or bid not in batter_stats.index:
            continue

        row = batter_stats.loc[bid]
        pa  = float(row.get("plate_appearances") or 0)
        if pa < 50:
            continue

        hr_rate   = float(row.get("hr_rate") or 0.0)
        if hr_rate <= 0:
            continue

        order    = batter["batting_order"]
        exp_pas  = _PA_BY_ORDER.get(order, _DEFAULT_PA)

        expected_hrs = hr_rate * exp_pas * pitcher_mod * park_factor * wind_mod * temp_mod

        # Binomial probability of at least 1 HR
        hr_prob = 1.0 - (1.0 - hr_rate) ** exp_pas

        results.append({
            "batter_name":     batter["name"] or str(row.get("batter_name", "")),
            "team":            str(row.get("team", "")),
            "batting_order":   order,
            "hr_prob":         round(hr_prob, 4),
            "expected_hrs":    round(expected_hrs, 4),
            "hr_rate_2026":    round(hr_rate, 5),
            "ab_per_hr":       float(row.get("ab_per_hr") or 0.0),
            "vs_pitcher_adj":  round(pitcher_mod, 3),
        })

    results.sort(key=lambda x: x["hr_prob"], reverse=True)
    return results


def predict_hr_props_for_pitcher(
    game_lineup: str | list | None,
    batter_stats: pd.DataFrame,
    pitcher_id: int | None,
    pitcher_stats: pd.DataFrame | None,
    weather: dict | None,
    park_factor: float = 1.0,
) -> list[dict]:
    """Like predict_hr_props but with an explicit pitcher for modifier calculation."""
    p_row = _pitcher_row(pitcher_id, pitcher_stats)
    pitcher_mod = 1.0
    if p_row is not None:
        hr9 = float(p_row.get("hr_per9") or 0.0)
        if hr9 > 0:
            pitcher_mod = hr9 / _LEAGUE_AVG_HR_PER9

    lineup = _parse_lineup(game_lineup)
    if not lineup or batter_stats.empty:
        return []

    wind_out = float((weather or {}).get("wind_out_factor", 0.0))
    temp_f   = float((weather or {}).get("temp_f", 72.0))
    wind_mod = 1.0 + (wind_out * 0.15)
    temp_mod = 1.0 + ((temp_f - 72.0) * 0.002)

    results = []
    for batter in lineup:
        bid = batter["id"]
        if bid == 0 or bid not in batter_stats.index:
            continue

        row = batter_stats.loc[bid]
        pa  = float(row.get("plate_appearances") or 0)
        if pa < 50:
            continue

        hr_rate_raw = row.get("hr_rate")
        import math as _math
        if hr_rate_raw is None or (isinstance(hr_rate_raw, float) and _math.isnan(hr_rate_raw)):
            continue
        hr_rate = float(hr_rate_raw)
        if hr_rate <= 0:
            continue

        order   = batter["batting_order"]
        exp_pas = _PA_BY_ORDER.get(order, _DEFAULT_PA)

        expected_hrs = hr_rate * exp_pas * pitcher_mod * park_factor * wind_mod * temp_mod
        hr_prob = 1.0 - (1.0 - hr_rate) ** exp_pas

        results.append({
            "batter_name":    batter["name"] or str(row.get("batter_name", "")),
            "team":           str(row.get("team", "")),
            "batting_order":  order,
            "hr_prob":        round(hr_prob, 4),
            "expected_hrs":   round(expected_hrs, 4),
            "hr_rate_2026":   round(hr_rate, 5),
            "ab_per_hr":      float(row.get("ab_per_hr") or 0.0),
            "vs_pitcher_adj": round(pitcher_mod, 3),
        })

    results.sort(key=lambda x: x["hr_prob"], reverse=True)
    return results


def predict_total_bases_props(
    game_lineup: str | list | None,
    batter_stats: pd.DataFrame,
    pitcher_id: int | None,
    pitcher_stats: pd.DataFrame | None,
    weather: dict | None = None,
) -> list[dict]:
    """Predict expected total bases props for each batter.

    Returns list ranked by expected_tbs descending, each entry with:
    batter_name, team, batting_order, expected_tbs, over_1_5, over_2_5.
    """
    lineup = _parse_lineup(game_lineup)
    if not lineup or batter_stats.empty:
        return []

    p_row = _pitcher_row(pitcher_id, pitcher_stats)
    pitcher_mod = 1.0
    if p_row is not None:
        h9 = float(p_row.get("hits_per9") or 0.0)
        if h9 > 0:
            pitcher_mod = h9 / _LEAGUE_AVG_HITS_PER9

    results = []
    for batter in lineup:
        bid = batter["id"]
        if bid == 0 or bid not in batter_stats.index:
            continue

        row = batter_stats.loc[bid]
        pa  = float(row.get("plate_appearances") or 0)
        if pa < 50:
            continue

        tb_rate = float(row.get("tb_rate") or 0.0)
        if tb_rate <= 0:
            continue

        order   = batter["batting_order"]
        exp_pas = _PA_BY_ORDER.get(order, _DEFAULT_PA)

        expected_tbs = tb_rate * exp_pas * pitcher_mod

        # Approximate P(TB >= line) using Poisson with lambda=expected_tbs
        lam = max(expected_tbs, 1e-6)
        over_1_5 = 1.0 - math.exp(-lam) * (1.0 + lam)  # P(X >= 2) = 1 - P(0) - P(1)
        # P(X >= 3) = 1 - P(0) - P(1) - P(2)
        over_2_5 = 1.0 - math.exp(-lam) * (1.0 + lam + lam**2 / 2.0)
        over_1_5 = max(0.0, min(1.0, over_1_5))
        over_2_5 = max(0.0, min(1.0, over_2_5))

        results.append({
            "batter_name":   batter["name"] or str(row.get("batter_name", "")),
            "team":          str(row.get("team", "")),
            "batting_order": order,
            "expected_tbs":  round(expected_tbs, 3),
            "over_1_5":      round(over_1_5, 4),
            "over_2_5":      round(over_2_5, 4),
        })

    results.sort(key=lambda x: x["expected_tbs"], reverse=True)
    return results


def predict_pitcher_k_props(
    pitcher_id: int | None,
    pitcher_stats: pd.DataFrame,
    opposing_lineup: str | list | None,
    batter_stats: pd.DataFrame,
) -> dict[str, Any]:
    """Predict strikeout props for a starting pitcher.

    Returns a dict with expected_ks, k_prob_per_pa, expected_innings,
    vs_lineup_k_rate, and prop_lines (over_4.5 through over_7.5).
    """
    empty: dict[str, Any] = {
        "pitcher_name":    "",
        "team":            "",
        "expected_ks":     0.0,
        "k_prob_per_pa":   0.0,
        "expected_innings": 0.0,
        "vs_lineup_k_rate": 0.0,
        "prop_lines": {
            "over_4.5": 0.0,
            "over_5.5": 0.0,
            "over_6.5": 0.0,
            "over_7.5": 0.0,
        },
    }

    if pitcher_id is None or pitcher_stats is None or pitcher_stats.empty:
        return empty

    if pitcher_id not in pitcher_stats.index:
        return empty

    p_row = pitcher_stats.loc[pitcher_id]

    pitcher_k_rate = float(p_row.get("k_rate") or 0.0)
    if pitcher_k_rate <= 0:
        # Fall back to k9 / (27 batters/9inn) approximation
        k9 = float(p_row.get("k9") or 0.0)
        pitcher_k_rate = k9 / 27.0 if k9 > 0 else 0.20

    innings_pitched = float(p_row.get("innings") or 0.0)
    games_started   = max(int(p_row.get("games_started") or 1), 1)
    expected_innings = min(6.0, innings_pitched / games_started)
    if expected_innings <= 0:
        expected_innings = 5.5

    # Opposing lineup K rate
    lineup = _parse_lineup(opposing_lineup)
    lineup_k_rate = 0.22  # league average fallback
    if lineup and not batter_stats.empty:
        k_rates = []
        for batter in lineup:
            bid = batter["id"]
            if bid != 0 and bid in batter_stats.index:
                kr = float(batter_stats.loc[bid].get("k_rate") or 0.0)
                if kr > 0:
                    k_rates.append(kr)
        if k_rates:
            lineup_k_rate = float(np.mean(k_rates))

    # Combined K probability per PA: geometric mean of pitcher and batter K rates
    k_prob_per_pa = math.sqrt(pitcher_k_rate * lineup_k_rate)

    expected_batters_faced = expected_innings * 3.3
    expected_ks = k_prob_per_pa * expected_batters_faced

    # Poisson CDF for prop lines
    lam = max(expected_ks, 1e-6)

    def _poisson_over(line: float) -> float:
        k_floor = int(math.floor(line))
        cumulative = sum(
            math.exp(-lam) * (lam ** k) / math.factorial(k)
            for k in range(k_floor + 1)
        )
        return round(max(0.0, min(1.0, 1.0 - cumulative)), 4)

    return {
        "pitcher_name":     str(p_row.get("pitcher_name", "")),
        "team":             str(p_row.get("team", "")),
        "expected_ks":      round(expected_ks, 2),
        "k_prob_per_pa":    round(k_prob_per_pa, 4),
        "expected_innings": round(expected_innings, 1),
        "vs_lineup_k_rate": round(lineup_k_rate, 4),
        "prop_lines": {
            "over_4.5": _poisson_over(4.5),
            "over_5.5": _poisson_over(5.5),
            "over_6.5": _poisson_over(6.5),
            "over_7.5": _poisson_over(7.5),
        },
    }

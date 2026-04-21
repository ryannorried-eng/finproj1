from __future__ import annotations

from line_tracker.bracket.simulator import REGIONS, ROUND_ORDER


def _all_teams() -> list[str]:
    return [team for region in REGIONS.values() for _, team in region]


def _blank_rounds() -> dict[str, float]:
    return {rnd: 0.0 for rnd in ROUND_ORDER}


PUBLIC_PICKS: dict[str, dict[str, float]] = {
    team: _blank_rounds()
    for team in _all_teams()
}

PUBLIC_PICKS.update({
    "Duke":        {"S16": 0.88, "E8": 0.69, "F4": 0.52, "Final": 0.33, "Champ": 0.22},
    "Michigan":    {"S16": 0.84, "E8": 0.58, "F4": 0.38, "Final": 0.20, "Champ": 0.11},
    "Arizona":     {"S16": 0.83, "E8": 0.57, "F4": 0.36, "Final": 0.19, "Champ": 0.10},
    "Florida":     {"S16": 0.86, "E8": 0.61, "F4": 0.45, "Final": 0.25, "Champ": 0.15},
    "Illinois":    {"S16": 0.72, "E8": 0.40, "F4": 0.18, "Final": 0.08, "Champ": 0.04},
    "Purdue":      {"S16": 0.77, "E8": 0.42, "F4": 0.20, "Final": 0.09, "Champ": 0.05},
    "Houston":     {"S16": 0.76, "E8": 0.41, "F4": 0.21, "Final": 0.10, "Champ": 0.06},
    "Iowa St.":    {"S16": 0.75, "E8": 0.39, "F4": 0.19, "Final": 0.09, "Champ": 0.05},
    "Connecticut": {"S16": 0.68, "E8": 0.27, "F4": 0.11, "Final": 0.04, "Champ": 0.02},
    "Gonzaga":     {"S16": 0.63, "E8": 0.24, "F4": 0.09, "Final": 0.03, "Champ": 0.01},
    "Michigan St.":{"S16": 0.58, "E8": 0.19, "F4": 0.07, "Final": 0.02, "Champ": 0.01},
    "Alabama":     {"S16": 0.52, "E8": 0.17, "F4": 0.08, "Final": 0.03, "Champ": 0.01},
    "Tennessee":   {"S16": 0.50, "E8": 0.16, "F4": 0.07, "Final": 0.02, "Champ": 0.01},
    "Texas Tech":  {"S16": 0.49, "E8": 0.15, "F4": 0.07, "Final": 0.02, "Champ": 0.01},
    "St. John's":  {"S16": 0.55, "E8": 0.20, "F4": 0.09, "Final": 0.03, "Champ": 0.01},
    "Kansas":      {"S16": 0.54, "E8": 0.19, "F4": 0.08, "Final": 0.03, "Champ": 0.01},
    "Louisville":  {"S16": 0.42, "E8": 0.12, "F4": 0.04, "Final": 0.01, "Champ": 0.00},
    "Wisconsin":   {"S16": 0.46, "E8": 0.13, "F4": 0.04, "Final": 0.01, "Champ": 0.00},
    "Vanderbilt":  {"S16": 0.43, "E8": 0.11, "F4": 0.04, "Final": 0.01, "Champ": 0.00},
    "Arkansas":    {"S16": 0.45, "E8": 0.13, "F4": 0.05, "Final": 0.01, "Champ": 0.00},
})


def normalize_public_picks(public_picks: dict[str, dict[str, float]] | None = None) -> dict[str, dict[str, float]]:
    src = public_picks or PUBLIC_PICKS
    out: dict[str, dict[str, float]] = {}

    for team in _all_teams():
        row = src.get(team, {})
        out[team] = {}
        for rnd in ROUND_ORDER:
            value = float(row.get(rnd, 0.0))
            if value < 0.0:
                value = 0.0
            if value > 1.0:
                value = 1.0
            out[team][rnd] = value

    return out

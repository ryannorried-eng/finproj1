from __future__ import annotations

import random

from line_tracker.bracket.simulator import REGIONS


def _pick_between(team_a: str, team_b: str, public_probs: dict, round_name: str, rng: random.Random) -> str:
    pa = float(public_probs.get(team_a, {}).get(round_name, 0.5))
    pb = float(public_probs.get(team_b, {}).get(round_name, 0.5))
    total = pa + pb
    p_team_a = 0.5 if total <= 0 else pa / total
    return team_a if rng.random() < p_team_a else team_b


def simulate_opponent_bracket(public_probs: dict, rng: random.Random) -> dict:
    out = {"R64": {}, "R32": {}, "S16": {}, "E8": {}, "Final": {}, "Champ": None}
    region_winners = {}

    for region_name, slot_list in REGIONS.items():
        teams = [team for _, team in slot_list]

        r64_pairs = [(teams[i], teams[i + 1]) for i in range(0, 16, 2)]
        r32 = []
        for idx, (a, b) in enumerate(r64_pairs, start=1):
            w = _pick_between(a, b, public_probs, "S16", rng)
            r32.append(w)
            out["R64"][f"{region_name}_G{idx}"] = w

        s16 = []
        for i in range(0, 8, 2):
            w = _pick_between(r32[i], r32[i + 1], public_probs, "E8", rng)
            s16.append(w)
            out["R32"][f"{region_name}_G{i//2 + 1}"] = w

        e8 = []
        for i in range(0, 4, 2):
            w = _pick_between(s16[i], s16[i + 1], public_probs, "F4", rng)
            e8.append(w)
            out["S16"][f"{region_name}_G{i//2 + 1}"] = w

        champ = _pick_between(e8[0], e8[1], public_probs, "Final", rng)
        out["E8"][f"{region_name}_G1"] = champ
        region_winners[region_name] = champ

    sf1 = _pick_between(region_winners["South"], region_winners["Midwest"], public_probs, "Final", rng)
    sf2 = _pick_between(region_winners["East"], region_winners["West"], public_probs, "Final", rng)
    out["Final"]["SF1"] = sf1
    out["Final"]["SF2"] = sf2
    out["Champ"] = _pick_between(sf1, sf2, public_probs, "Champ", rng)
    return out

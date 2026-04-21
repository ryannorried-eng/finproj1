from __future__ import annotations

import random

from line_tracker.bracket.simulator import TrainedBracketSimulator, REGIONS


def simulate_single_tournament(sim: TrainedBracketSimulator, rng: random.Random) -> dict:
    out = {"R64": {}, "R32": {}, "S16": {}, "E8": {}, "Final": {}, "Champ": None}
    region_winners = {}

    for region_name, slot_list in REGIONS.items():
        teams = [team for _, team in slot_list]

        r64_pairs = [(teams[i], teams[i + 1]) for i in range(0, 16, 2)]
        r32 = []
        for idx, (a, b) in enumerate(r64_pairs, start=1):
            w = sim.simulate_game(a, b, rng)
            r32.append(w)
            out["R64"][f"{region_name}_G{idx}"] = w

        s16 = []
        for i in range(0, 8, 2):
            w = sim.simulate_game(r32[i], r32[i + 1], rng)
            s16.append(w)
            out["R32"][f"{region_name}_G{i//2 + 1}"] = w

        e8 = []
        for i in range(0, 4, 2):
            w = sim.simulate_game(s16[i], s16[i + 1], rng)
            e8.append(w)
            out["S16"][f"{region_name}_G{i//2 + 1}"] = w

        champ = sim.simulate_game(e8[0], e8[1], rng)
        out["E8"][f"{region_name}_G1"] = champ
        region_winners[region_name] = champ

    sf1 = sim.simulate_game(region_winners["South"], region_winners["Midwest"], rng)
    sf2 = sim.simulate_game(region_winners["East"], region_winners["West"], rng)
    out["Final"]["SF1"] = sf1
    out["Final"]["SF2"] = sf2
    out["Champ"] = sim.simulate_game(sf1, sf2, rng)
    return out

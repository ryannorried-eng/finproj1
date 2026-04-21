from __future__ import annotations

from copy import deepcopy

from line_tracker.bracket.simulator import REGIONS, TrainedBracketSimulator
from line_tracker.bracket.variants import suggest_all_region_pivots


def _public_round(public_picks: dict[str, dict[str, float]], team: str, round_name: str) -> float:
    row = public_picks.get(team, {})
    if round_name in row:
        return float(row.get(round_name, 0.0))
    if round_name == "R32":
        return float(row.get("S16", 0.0))
    if round_name == "S16":
        return float(row.get("E8", 0.0))
    return 0.0


def _region_from_overrides(
    sim: TrainedBracketSimulator,
    region_name: str,
    overrides: dict[str, dict[str, str]] | None = None,
) -> dict:
    overrides = overrides or {}
    out = {"R64": {}, "R32": {}, "S16": {}, "E8": {}}

    teams = [team for _, team in REGIONS[region_name]]

    r64_pairs = [(teams[i], teams[i + 1]) for i in range(0, 16, 2)]
    r32 = []
    for idx, (a, b) in enumerate(r64_pairs, start=1):
        key = f"{region_name}_G{idx}"
        forced = overrides.get("R64", {}).get(key)
        if forced in {a, b}:
            w = forced
        else:
            w, _, _ = sim.deterministic_game(a, b)
        r32.append(w)
        out["R64"][key] = w

    s16 = []
    for i in range(0, 8, 2):
        a, b = r32[i], r32[i + 1]
        key = f"{region_name}_G{i//2 + 1}"
        forced = overrides.get("R32", {}).get(key)
        if forced in {a, b}:
            w = forced
        else:
            w, _, _ = sim.deterministic_game(a, b)
        s16.append(w)
        out["R32"][key] = w

    e8 = []
    for i in range(0, 4, 2):
        a, b = s16[i], s16[i + 1]
        key = f"{region_name}_G{i//2 + 1}"
        forced = overrides.get("S16", {}).get(key)
        if forced in {a, b}:
            w = forced
        else:
            w, _, _ = sim.deterministic_game(a, b)
        e8.append(w)
        out["S16"][key] = w

    a, b = e8[0], e8[1]
    key = f"{region_name}_G1"
    forced = overrides.get("E8", {}).get(key)
    if forced in {a, b}:
        champ = forced
    else:
        champ, _, _ = sim.deterministic_game(a, b)
    out["E8"][key] = champ

    return out


def _build_full_bracket(
    sim: TrainedBracketSimulator,
    overrides: dict[str, dict[str, str]] | None = None,
) -> dict:
    overrides = overrides or {}

    out = {"R64": {}, "R32": {}, "S16": {}, "E8": {}, "Final": {}, "Champ": None}
    region_winners = {}

    for region_name in REGIONS:
        region = _region_from_overrides(sim, region_name, overrides)
        out["R64"].update(region["R64"])
        out["R32"].update(region["R32"])
        out["S16"].update(region["S16"])
        out["E8"].update(region["E8"])
        region_winners[region_name] = region["E8"][f"{region_name}_G1"]

    sf1, _, _ = sim.deterministic_game(region_winners["South"], region_winners["Midwest"])
    sf2, _, _ = sim.deterministic_game(region_winners["East"], region_winners["West"])
    out["Final"]["SF1"] = sf1
    out["Final"]["SF2"] = sf2

    champ, _, _ = sim.deterministic_game(sf1, sf2)
    out["Champ"] = champ
    return out


def build_deterministic_full_bracket(sim: TrainedBracketSimulator) -> dict:
    return _build_full_bracket(sim, overrides={})


def _set_champion(bracket: dict, champ_team: str) -> dict:
    b = deepcopy(bracket)
    if champ_team in {b["Final"].get("SF1"), b["Final"].get("SF2")}:
        b["Champ"] = champ_team
    return b


def suggest_early_round_pivots(
    sim: TrainedBracketSimulator,
    base_bracket: dict,
    public_picks: dict[str, dict[str, float]],
) -> list[dict]:
    suggestions = []

    for region_name, slot_list in REGIONS.items():
        teams = [team for _, team in slot_list]
        r64_pairs = [(teams[i], teams[i + 1]) for i in range(0, 16, 2)]

        for idx, (a, b) in enumerate(r64_pairs, start=1):
            key = f"{region_name}_G{idx}"
            base_pick = base_bracket["R64"][key]
            alt_pick = b if base_pick == a else a

            _, p_a = sim.predict_neutral_matchup(a, b)
            p = {a: p_a, b: 1.0 - p_a}

            base_edge = p[base_pick] - _public_round(public_picks, base_pick, "R32")
            alt_edge = p[alt_pick] - _public_round(public_picks, alt_pick, "R32")

            if alt_edge > base_edge:
                suggestions.append({
                    "round": "R64",
                    "region": region_name,
                    "game_key": key,
                    "out": base_pick,
                    "in": alt_pick,
                    "base_edge": base_edge,
                    "pivot_edge": alt_edge,
                    "gain": alt_edge - base_edge,
                    "model_prob_in": p[alt_pick],
                    "public_prob_in": _public_round(public_picks, alt_pick, "R32"),
                })

    for region_name in REGIONS:
        g1 = base_bracket["R64"][f"{region_name}_G1"]
        g2 = base_bracket["R64"][f"{region_name}_G2"]
        g3 = base_bracket["R64"][f"{region_name}_G3"]
        g4 = base_bracket["R64"][f"{region_name}_G4"]
        g5 = base_bracket["R64"][f"{region_name}_G5"]
        g6 = base_bracket["R64"][f"{region_name}_G6"]
        g7 = base_bracket["R64"][f"{region_name}_G7"]
        g8 = base_bracket["R64"][f"{region_name}_G8"]

        r32_games = [
            (f"{region_name}_G1", g1, g2),
            (f"{region_name}_G2", g3, g4),
            (f"{region_name}_G3", g5, g6),
            (f"{region_name}_G4", g7, g8),
        ]

        for key, a, b in r32_games:
            base_pick = base_bracket["R32"][key]
            alt_pick = b if base_pick == a else a

            _, p_a = sim.predict_neutral_matchup(a, b)
            p = {a: p_a, b: 1.0 - p_a}

            base_edge = p[base_pick] - _public_round(public_picks, base_pick, "S16")
            alt_edge = p[alt_pick] - _public_round(public_picks, alt_pick, "S16")

            if alt_edge > base_edge:
                suggestions.append({
                    "round": "R32",
                    "region": region_name,
                    "game_key": key,
                    "out": base_pick,
                    "in": alt_pick,
                    "base_edge": base_edge,
                    "pivot_edge": alt_edge,
                    "gain": alt_edge - base_edge,
                    "model_prob_in": p[alt_pick],
                    "public_prob_in": _public_round(public_picks, alt_pick, "S16"),
                })

    suggestions.sort(key=lambda x: x["gain"], reverse=True)
    return suggestions


def _apply_single_override(base_overrides: dict, pivot: dict) -> dict:
    overrides = deepcopy(base_overrides)
    overrides.setdefault(pivot["round"], {})
    overrides[pivot["round"]][pivot["game_key"]] = pivot["in"]
    return overrides


def build_candidate_brackets(
    sim: TrainedBracketSimulator,
    model_probs: dict[str, dict[str, float]],
    public_picks: dict[str, dict[str, float]],
) -> dict[str, dict]:
    base = build_deterministic_full_bracket(sim)
    candidates: dict[str, dict] = {"accuracy": base}

    late_pivots = suggest_all_region_pivots(model_probs, public_picks)

    if len(late_pivots) >= 1:
        p = late_pivots[0]
        overrides = {"E8": {f"{p['region']}_G1": p["pivot"]}}
        candidates["medium"] = _build_full_bracket(sim, overrides)

    if len(late_pivots) >= 2:
        overrides = {
            "E8": {
                f"{late_pivots[0]['region']}_G1": late_pivots[0]["pivot"],
                f"{late_pivots[1]['region']}_G1": late_pivots[1]["pivot"],
            }
        }
        candidates["aggressive"] = _build_full_bracket(sim, overrides)

    early_pivots = suggest_early_round_pivots(sim, base, public_picks)

    if len(early_pivots) >= 1:
        overrides = _apply_single_override({}, early_pivots[0])
        candidates["early1"] = _build_full_bracket(sim, overrides)

    if len(early_pivots) >= 2:
        overrides = {}
        used_games = set()
        for pivot in early_pivots:
            if pivot["game_key"] in used_games:
                continue
            overrides = _apply_single_override(overrides, pivot)
            used_games.add(pivot["game_key"])
            if len(used_games) == 2:
                break
        candidates["early2"] = _build_full_bracket(sim, overrides)

    if len(late_pivots) >= 1 and len(early_pivots) >= 1:
        overrides = {"E8": {f"{late_pivots[0]['region']}_G1": late_pivots[0]["pivot"]}}
        overrides = _apply_single_override(overrides, early_pivots[0])
        candidates["hybrid"] = _build_full_bracket(sim, overrides)

    extras: dict[str, dict] = {}
    for name, bracket in list(candidates.items()):
        for champ_team in (bracket["Final"].get("SF1"), bracket["Final"].get("SF2")):
            if champ_team:
                key = f"{name}_champ_{champ_team.replace(' ', '_').replace('.', '')}"
                extras[key] = _set_champion(bracket, champ_team)

    candidates.update(extras)
    return candidates

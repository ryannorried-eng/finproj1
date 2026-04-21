from __future__ import annotations

from line_tracker.bracket.simulator import REGIONS, TrainedBracketSimulator


def deterministic_region_winner(sim: TrainedBracketSimulator, region_name: str) -> str:
    teams = [team for _, team in REGIONS[region_name]]

    r64_pairs = [(teams[i], teams[i + 1]) for i in range(0, 16, 2)]
    r32 = []
    for a, b in r64_pairs:
        w, _, _ = sim.deterministic_game(a, b)
        r32.append(w)

    s16 = []
    for i in range(0, 8, 2):
        w, _, _ = sim.deterministic_game(r32[i], r32[i + 1])
        s16.append(w)

    e8 = []
    for i in range(0, 4, 2):
        w, _, _ = sim.deterministic_game(s16[i], s16[i + 1])
        e8.append(w)

    champ, _, _ = sim.deterministic_game(e8[0], e8[1])
    return champ


def build_accuracy_final_four(sim: TrainedBracketSimulator) -> dict[str, str]:
    return {
        region_name: deterministic_region_winner(sim, region_name)
        for region_name in REGIONS
    }


def suggest_region_pivot(
    region_name: str,
    model_probs: dict[str, dict[str, float]],
    public_picks: dict[str, dict[str, float]],
    min_model_f4: float = 0.12,
) -> dict | None:
    teams = [team for _, team in REGIONS[region_name]]

    favorite = max(
        teams,
        key=lambda t: model_probs.get(t, {}).get("F4", 0.0)
    )

    favorite_edge = (
        model_probs.get(favorite, {}).get("F4", 0.0)
        - public_picks.get(favorite, {}).get("F4", 0.0)
    )

    candidates = []
    for team in teams:
        if team == favorite:
            continue

        model_f4 = model_probs.get(team, {}).get("F4", 0.0)
        public_f4 = public_picks.get(team, {}).get("F4", 0.0)
        edge = model_f4 - public_f4

        if model_f4 >= min_model_f4:
            candidates.append((edge, model_f4, team, public_f4))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_edge, best_model_f4, best_team, best_public_f4 = candidates[0]

    if best_edge <= favorite_edge:
        return None

    return {
        "region": region_name,
        "favorite": favorite,
        "favorite_model_f4": model_probs.get(favorite, {}).get("F4", 0.0),
        "favorite_public_f4": public_picks.get(favorite, {}).get("F4", 0.0),
        "favorite_edge": favorite_edge,
        "pivot": best_team,
        "pivot_model_f4": best_model_f4,
        "pivot_public_f4": best_public_f4,
        "pivot_edge": best_edge,
    }


def suggest_all_region_pivots(
    model_probs: dict[str, dict[str, float]],
    public_picks: dict[str, dict[str, float]],
) -> list[dict]:
    pivots = []
    for region_name in REGIONS:
        pivot = suggest_region_pivot(region_name, model_probs, public_picks)
        if pivot:
            pivots.append(pivot)
    pivots.sort(key=lambda x: x["pivot_edge"] - x["favorite_edge"], reverse=True)
    return pivots

from __future__ import annotations

ROUND_SCORES = {
    "R64": 1,
    "R32": 2,
    "S16": 4,
    "E8": 8,
    "Final": 16,
    "Champ": 32,
}


def score_bracket(actual: dict, bracket: dict) -> int:
    score = 0

    for round_name in ("R64", "R32", "S16", "E8"):
        actual_round = actual.get(round_name, {})
        bracket_round = bracket.get(round_name, {})
        for game_key, actual_winner in actual_round.items():
            if bracket_round.get(game_key) == actual_winner:
                score += ROUND_SCORES[round_name]

    actual_final = actual.get("Final", {})
    bracket_final = bracket.get("Final", {})
    for game_key, actual_winner in actual_final.items():
        if bracket_final.get(game_key) == actual_winner:
            score += ROUND_SCORES["Final"]

    if bracket.get("Champ") == actual.get("Champ"):
        score += ROUND_SCORES["Champ"]

    return score

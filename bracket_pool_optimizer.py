from __future__ import annotations

import random
from collections import defaultdict

from line_tracker.bracket.simulator import run_trained_bracket_simulation
from line_tracker.bracket.public_picks import normalize_public_picks
from line_tracker.bracket.candidates import build_candidate_brackets
from line_tracker.bracket.field_simulator import simulate_opponent_bracket
from line_tracker.bracket.tournament_runner import simulate_single_tournament
from line_tracker.bracket.scorer import score_bracket


N_TOURNEY_SIMS = 2000
N_OPPONENTS = 50


def main():
    result = run_trained_bracket_simulation(n_sims=5000)
    sim = result["sim"]
    model_probs = result["team_round_probs"]
    public_picks = normalize_public_picks()

    candidates = build_candidate_brackets(sim, model_probs, public_picks)
    print("loaded candidates:", ", ".join(candidates.keys()))

    rng = random.Random(42)
    first_equity = defaultdict(float)
    avg_score = defaultdict(float)

    for _ in range(N_TOURNEY_SIMS):
        actual = simulate_single_tournament(sim, rng)

        scores = {}
        for name, bracket in candidates.items():
            scores[name] = score_bracket(actual, bracket)
            avg_score[name] += scores[name]

        for i in range(N_OPPONENTS):
            opp = simulate_opponent_bracket(public_picks, rng)
            scores[f"opp_{i}"] = score_bracket(actual, opp)

        best_score = max(scores.values())
        winners = [name for name, s in scores.items() if s == best_score]
        candidate_winners = [w for w in winners if w in candidates]
        if candidate_winners:
            share = 1.0 / len(winners)
            for w in candidate_winners:
                first_equity[w] += share

    print("\n" + "=" * 68)
    print("  POOL OPTIMIZER — FULL BRACKET SCORING")
    print("=" * 68)

    ranked = sorted(
        candidates.keys(),
        key=lambda name: (first_equity[name] / N_TOURNEY_SIMS, avg_score[name] / N_TOURNEY_SIMS),
        reverse=True,
    )

    for name in ranked:
        win_pct = first_equity[name] / N_TOURNEY_SIMS
        mean_score = avg_score[name] / N_TOURNEY_SIMS
        champ = candidates[name].get("Champ")
        ff = candidates[name].get("E8", {})
        ff_summary = ", ".join(f"{k.replace('_G1','')}={v}" for k, v in ff.items())
        print(f"{name:<30} first_equity={win_pct:6.3f}  avg_score={mean_score:6.2f}  champ={champ}")
        print(f"  {ff_summary}")

    print("\nBest current bracket:", ranked[0])


if __name__ == "__main__":
    main()

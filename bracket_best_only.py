from __future__ import annotations

from line_tracker.bracket.simulator import run_trained_bracket_simulation
from line_tracker.bracket.public_picks import normalize_public_picks
from line_tracker.bracket.candidates import build_candidate_brackets


BEST_NAME = "hybrid_champ_Arizona"


def print_round(title: str, picks):
    print(f"\n{title}")
    print("-" * len(title))
    if isinstance(picks, dict):
        for k, v in picks.items():
            print(f"  {k:<18} {v}")
    else:
        print(f"  {picks}")


def main():
    result = run_trained_bracket_simulation(n_sims=3000)
    candidates = build_candidate_brackets(
        result["sim"],
        result["team_round_probs"],
        normalize_public_picks(),
    )

    bracket = candidates[BEST_NAME]

    print("\n" + "=" * 78)
    print(BEST_NAME.upper())
    print("=" * 78)

    print_round("ROUND OF 64 WINNERS", bracket["R64"])
    print_round("ROUND OF 32 WINNERS", bracket["R32"])
    print_round("SWEET 16 WINNERS", bracket["S16"])
    print_round("ELITE 8 / FINAL FOUR TEAMS", bracket["E8"])
    print_round("FINAL FOUR / TITLE GAME TEAMS", bracket["Final"])
    print_round("CHAMPION", bracket["Champ"])


if __name__ == "__main__":
    main()

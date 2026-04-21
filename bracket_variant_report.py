from __future__ import annotations

from line_tracker.bracket.simulator import run_trained_bracket_simulation
from line_tracker.bracket.public_picks import normalize_public_picks
from line_tracker.bracket.variants import (
    build_accuracy_final_four,
    suggest_all_region_pivots,
)


def print_block(title: str, data: dict[str, str]):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    for region, team in data.items():
        print(f"  {region:<10} {team}")


def main():
    result = run_trained_bracket_simulation(n_sims=10000)
    sim = result["sim"]
    model_probs = result["team_round_probs"]
    public_picks = normalize_public_picks()

    accuracy = build_accuracy_final_four(sim)

    print_block("ACCURACY FINAL FOUR", accuracy)

    print("\n" + "=" * 72)
    print("  REGION PIVOT SUGGESTIONS")
    print("=" * 72)
    for p in suggest_all_region_pivots(model_probs, public_picks):
        print(
            f"  {p['region']:<10} out={p['favorite']:<14} in={p['pivot']:<14} "
            f"fav_edge={100*p['favorite_edge']:5.1f}%  pivot_edge={100*p['pivot_edge']:5.1f}%"
        )


if __name__ == "__main__":
    main()

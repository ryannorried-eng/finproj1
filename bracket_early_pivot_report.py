from __future__ import annotations

from line_tracker.bracket.simulator import run_trained_bracket_simulation
from line_tracker.bracket.public_picks import normalize_public_picks
from line_tracker.bracket.candidates import build_deterministic_full_bracket, suggest_early_round_pivots


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def main():
    result = run_trained_bracket_simulation(n_sims=3000)
    sim = result["sim"]
    public_picks = normalize_public_picks()
    base = build_deterministic_full_bracket(sim)

    pivots = suggest_early_round_pivots(sim, base, public_picks)

    print("\n" + "=" * 78)
    print("EARLY ROUND LEVERAGE PIVOTS")
    print("=" * 78)

    for p in pivots[:20]:
        print(
            f"{p['round']:<4}  {p['game_key']:<12} "
            f"out={p['out']:<14} in={p['in']:<14} "
            f"gain={fmt_pct(p['gain'])}  "
            f"model_in={fmt_pct(p['model_prob_in'])}  "
            f"public_in={fmt_pct(p['public_prob_in'])}"
        )


if __name__ == "__main__":
    main()

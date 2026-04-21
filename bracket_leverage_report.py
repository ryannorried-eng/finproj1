from __future__ import annotations

from line_tracker.bracket.simulator import run_trained_bracket_simulation
from line_tracker.bracket.public_picks import normalize_public_picks
from line_tracker.bracket.leverage import (
    compute_leverage_rows,
    filter_round,
    sort_by_edge,
    sort_by_weighted_edge,
)


def print_section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def print_rows(rows: list[dict], limit: int = 12):
    for row in rows[:limit]:
        print(
            f"  {row['team']:<20}"
            f" model={fmt_pct(row['model_prob'])}  "
            f"public={fmt_pct(row['public_prob'])}  "
            f"edge={fmt_pct(row['edge'])}"
        )


def main():
    result = run_trained_bracket_simulation(n_sims=10000)
    model_probs = result["team_round_probs"]
    public_picks = normalize_public_picks()
    rows = compute_leverage_rows(model_probs, public_picks)

    print_section("TOP SWEET 16 LEVERAGE")
    print_rows(sort_by_edge(filter_round(rows, "S16")), limit=12)

    print_section("TOP ELITE EIGHT LEVERAGE")
    print_rows(sort_by_edge(filter_round(rows, "E8")), limit=12)

    print_section("TOP FINAL FOUR LEVERAGE")
    print_rows(sort_by_edge(filter_round(rows, "F4")), limit=12)

    print_section("TOP TITLE GAME LEVERAGE")
    print_rows(sort_by_edge(filter_round(rows, "Final")), limit=12)

    print_section("TOP CHAMPION LEVERAGE")
    print_rows(sort_by_edge(filter_round(rows, "Champ")), limit=12)

    print_section("MOST OVEROWNED FINAL FOUR TEAMS")
    print_rows(sort_by_edge(filter_round(rows, "F4"), descending=False), limit=12)

    print_section("MOST OVEROWNED CHAMPION TEAMS")
    print_rows(sort_by_edge(filter_round(rows, "Champ"), descending=False), limit=12)

    print_section("BEST TOTAL WEIGHTED LEVERAGE")
    totals = [r for r in rows if r["round"] == "TOTAL"]
    for row in sort_by_weighted_edge(totals)[:15]:
        print(f"  {row['team']:<20} weighted_edge={row['weighted_edge']:6.3f}")


if __name__ == "__main__":
    main()

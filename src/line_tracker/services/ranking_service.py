"""Step 4 – Ranking-first betting: top tail only.

After pruning, rank by a composite score and take only the top N picks.
"""

from __future__ import annotations


def _sort_key(entry: dict) -> tuple:
    """Composite sort key: (edge_z desc, edge_ev_shrunk desc, quality desc)."""
    return (
        -(entry.get("edge_z") or 0),
        -(entry.get("edge_ev_shrunk") or 0),
        -(entry.get("quality_score") or 0),
    )


def select_top_picks(
    pruned: list[dict],
    *,
    top_n: int = 3,
) -> list[dict]:
    """Sort pruned entries by composite score and return top *top_n*."""
    ranked = sorted(pruned, key=_sort_key)
    return ranked[:top_n]


def format_picks_report(picks: list[dict]) -> str:
    """Format a human-readable CLI report of selected picks."""
    if not picks:
        return "No actionable picks survived pruning."

    lines: list[str] = [f"=== Top {len(picks)} Picks ===", ""]
    for i, p in enumerate(picks, 1):
        lines.append(
            f"  #{i}  {p.get('market', '?')} {p.get('selection', '?')}"
            f"  @{p.get('best_sportsbook', '?')}"
        )
        lines.append(
            f"      Edge-Z: {p.get('edge_z', 0):.2f}"
            f"  |  EV-shrunk: {(p.get('edge_ev_shrunk') or 0) * 100:.1f}%"
            f"  |  Quality: {p.get('quality_score', 0)}"
        )
        lines.append(
            f"      Odds: {p.get('best_odds', 0):+.0f}"
            f"  |  Consensus: {p.get('consensus_prob', 0):.3f}"
            f"  |  Tier: {p.get('tier', '?')}"
            f"  |  Alpha: {p.get('alpha_label', '?')}"
        )
        lines.append("")
    return "\n".join(lines)

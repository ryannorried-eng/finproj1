"""Step 4 – Ranking-first betting: top tail only.

After pruning, rank by a composite score and take only the top N picks.
"""

from __future__ import annotations


def _sort_key(entry: dict) -> tuple:
    """Composite sort key — model entries rank by edge_pct, consensus by edge_z."""
    if entry.get("edge_source") == "model":
        return (
            -(entry.get("edge_pct") or 0),
            -(entry.get("edge_ev_shrunk") or 0),
            -(entry.get("quality_score") or 0),
            -(entry.get("books_used") or 0),
        )
    return (
        -(entry.get("edge_z") or 0),
        -(entry.get("edge_ev_shrunk") or 0),
        -(entry.get("quality_score") or 0),
        -(entry.get("books_used") or 0),
    )


def select_top_picks(
    pruned: list[dict],
    *,
    top_n: int = 5,
) -> list[dict]:
    """Sort candidates by composite score and return top *top_n*.

    Always returns up to *top_n* entries if candidates exist — returns 0
    only when there are literally no valid entries.
    """
    ranked = sorted(pruned, key=_sort_key)
    return ranked[:top_n]


def format_picks_report(picks: list[dict]) -> str:
    """Format a human-readable CLI report of selected picks."""
    if not picks:
        return "No actionable picks available."

    lines: list[str] = [f"=== Best {len(picks)} Available Picks ===", ""]
    for i, p in enumerate(picks, 1):
        lines.append(
            f"  #{i}  {p.get('market', '?')} {p.get('selection', '?')}"
            f"  @{p.get('best_sportsbook', '?')}"
        )
        is_model = p.get("edge_source") == "model"
        if is_model:
            _mp = p.get("model_prob")
            _mktp = p.get("market_prob")
            mp_str = f"{_mp:.3f}" if _mp is not None else "N/A"
            mktp_str = f"{_mktp:.3f}" if _mktp is not None else "N/A"
            lines.append(
                f"      Edge%: {p.get('edge_pct', 0):+.1f}"
                f"  |  EV-shrunk: {(p.get('edge_ev_shrunk') or 0) * 100:.1f}%"
                f"  |  Model: {mp_str} vs Market: {mktp_str}"
            )
        else:
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

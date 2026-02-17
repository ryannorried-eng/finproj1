"""Explainability UI component for debug-mode pick explanations.

Renders a 'Why this pick?' expander with explanation summary, key metrics
table, and optional raw JSON — for both Daily Slate and Best Lines pages.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from line_tracker.models import BestBetResult
from line_tracker.services.explanation_service import format_explanation_summary

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_pick_explanation(
    entry: dict[str, Any],
    *,
    debug_enabled: bool,
    context: str = "slate",
) -> None:
    """Render a 'Why this pick?' expander for a single recommendation.

    Parameters
    ----------
    entry:
        A slate entry dict (Daily Slate) or standout dict (Best Lines).
        For Daily Slate entries, ``entry["best_bet_result"]`` should be a
        :class:`BestBetResult` instance.  For Best Lines standouts the
        available shopping fields are used instead.
    debug_enabled:
        Value of ``st.session_state["diag_debug_mode"]``.
        When False this function is a no-op (no computation, no UI).
    context:
        ``"slate"`` for Daily Slate entries, ``"shopping"`` for Best Lines
        standouts.
    """
    if not debug_enabled:
        return

    if context == "shopping":
        _render_shopping_explanation(entry)
    else:
        _render_slate_explanation(entry)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_slate_explanation(entry: dict[str, Any]) -> None:
    """Render explainability for a Daily Slate entry backed by BestBetResult."""
    bbr: BestBetResult | None = entry.get("best_bet_result")

    label = (
        f"Why this pick? — {entry.get('selection', '?')} "
        f"({entry.get('market', '?')})"
    )
    with st.expander(label, expanded=False):
        # (a) Human-readable summary
        if bbr is not None:
            st.markdown(f"```\n{format_explanation_summary(bbr)}\n```")
        else:
            st.caption("No BestBetResult attached to this entry.")

        # (b) Key metrics table
        st.markdown("**Key Metrics**")
        _render_metrics_table(entry, bbr)

        # (c) Raw JSON toggle
        if bbr is not None and bbr.explanation:
            if st.checkbox(
                "Show raw explanation JSON",
                value=False,
                key=(
                    f"raw_json_{entry.get('event_id', entry.get('event', ''))}"
                    f"_{entry.get('market', '')}_{entry.get('selection', '')}"
                ),
            ):
                st.json(bbr.explanation)


def _render_shopping_explanation(entry: dict[str, Any]) -> None:
    """Render explainability for a Best Lines standout dict."""
    label = (
        f"Why this pick? — {entry.get('selection', '?')} "
        f"({entry.get('market', '?')})"
    )
    with st.expander(label, expanded=False):
        # Summary
        edge = entry.get("edge", 0.0)
        st.markdown(
            f"**Edge:** {edge * 100:.2f}% vs consensus  \n"
            f"**Consensus prob (excl. book):** "
            f"{entry.get('consensus_prob', 0.0):.4f}  \n"
            f"**Book implied prob:** {entry.get('book_prob', 0.0):.4f}  \n"
            f"**Sportsbook:** {entry.get('sportsbook', 'N/A')}  \n"
            f"**Odds:** {entry.get('odds', 'N/A')} "
            f"(median: {entry.get('median_odds', 'N/A')})"
        )

        # Shopping-context metrics
        st.markdown("**Shopping Context**")
        metrics: dict[str, str] = {
            "Books used (excl)": str(entry.get("books_used_excl", "N/A")),
            "Consensus method": str(entry.get("consensus_method", "N/A")),
            "Dollar impact": f"${entry.get('dollar_impact', 0.0):+.2f}",
            "Execution adv/$100": f"${entry.get('exec_adv_100', 0.0):+.2f}",
            "d_best": f"{entry.get('d_best', 0.0):.4f}",
            "d_ref": f"{entry.get('d_ref', 0.0):.4f}",
        }

        rows = [
            f"| {k} | {v} |" for k, v in metrics.items()
        ]
        table = "| Metric | Value |\n|---|---|\n" + "\n".join(rows)
        st.markdown(table)


def _render_metrics_table(
    entry: dict[str, Any],
    bbr: BestBetResult | None,
) -> None:
    """Render the key metrics table for a slate entry."""
    # Prefer BestBetResult fields when available, fall back to entry dict.
    def _val(bbr_attr: str, entry_key: str, fmt: str = "") -> str:
        v = None
        if bbr is not None:
            v = getattr(bbr, bbr_attr, None)
        if v is None or v == "":
            v = entry.get(entry_key)
        if v is None:
            return "N/A"
        if fmt:
            return format(v, fmt)
        return str(v)

    metrics: dict[str, str] = {
        "Edge %": _val("edge_pct", "edge_pct", "+.2f"),
        "Consensus prob": _val("consensus_prob", "consensus_prob", ".4f"),
        "Best odds": _val("best_odds_american", "best_odds", "+.0f"),
        "Best book": _val("best_sportsbook", "best_sportsbook"),
        "Quality tier": _val("quality_tier", "quality_tier"),
        "Quality score": _val("quality_score", "quality_score"),
        "Recency weight": _val("recency_weight", "recency_weight", ".4f"),
        "Volatility sigma": _val("volatility_sigma", "market_volatility_sigma", ".4f"),
        "Outliers removed": _val("outliers_removed", "outliers_removed"),
        "Kelly suggested": _val("kelly_suggested", "kelly_suggested", ".4f"),
        "Sizing note": _val("sizing_note", "sizing_note"),
    }

    rows = [f"| {k} | {v} |" for k, v in metrics.items()]
    table = "| Metric | Value |\n|---|---|\n" + "\n".join(rows)
    st.markdown(table)

"""Explainability UI component for Debug mode pick explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from line_tracker.models import BestBetResult
from line_tracker.services.explanation_service import format_explanation_summary


def render_pick_explanation(entry: dict[str, Any], debug_enabled: bool) -> None:
    """Render a 'Why this pick?' expander when debug mode is active.

    Parameters
    ----------
    entry:
        A slate entry dict (Daily Slate) or shopping standout dict
        (Best Lines to Shop).  If the entry carries a ``best_bet_result``
        key with a :class:`BestBetResult`, the full explanation is shown.
        Otherwise a lighter shopping-context view is rendered.
    debug_enabled:
        Value of ``st.session_state.get("diag_debug_mode", False)``.
        When *False* the function returns immediately with no output.
    """
    if not debug_enabled:
        return

    bbr: BestBetResult | None = entry.get("best_bet_result")  # type: ignore[assignment]

    if bbr is not None:
        _render_bbr_explanation(bbr, entry)
    else:
        _render_shopping_explanation(entry)


# ── Internals ──────────────────────────────────────────────────────────


def _render_bbr_explanation(bbr: BestBetResult, entry: dict[str, Any]) -> None:
    """Full explanation backed by a BestBetResult (Daily Slate context)."""
    with st.expander("Why this pick?", expanded=False):
        # (a) Human-readable summary
        st.markdown("**Explanation summary**")
        st.text(format_explanation_summary(bbr))

        st.markdown("---")

        # (b) Key metrics table
        st.markdown("**Key metrics**")
        metrics = _build_metrics_table(bbr, entry)
        st.dataframe(
            pd.DataFrame(metrics, columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
        )

        # (c) Raw JSON toggle — only built when checked
        if st.checkbox(
            "Show raw explanation JSON",
            value=False,
            key=f"_raw_json_{entry.get('event_id', '')}"
                f"_{bbr.market}_{bbr.side}",
        ):
            st.json(bbr.explanation)


def _render_shopping_explanation(entry: dict[str, Any]) -> None:
    """Lighter explanation for Best Lines to Shop standout dicts."""
    with st.expander("Why this pick?", expanded=False):
        st.markdown("**Shopping context**")
        metrics: list[list[str]] = [
            ["Edge (%)", f"{entry.get('edge', 0) * 100:.2f}%"],
            ["Consensus prob", f"{entry.get('consensus_prob', 0):.4f}"],
            ["Book prob", f"{entry.get('book_prob', 0):.4f}"],
            ["Dollar impact", f"${entry.get('dollar_impact', 0):+.2f}"],
            ["Exec advantage/$100", f"${entry.get('exec_adv_100', 0):+.2f}"],
        ]

        books_excl = entry.get("books_used_excl")
        if books_excl is not None:
            metrics.append(["Books used (excl)", str(books_excl)])

        method = entry.get("consensus_method", "")
        if method:
            metrics.append(["Consensus method", method])

        st.dataframe(
            pd.DataFrame(metrics, columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
        )


def _build_metrics_table(
    bbr: BestBetResult,
    entry: dict[str, Any],
) -> list[list[str]]:
    """Build a list of [metric, value] pairs for the key-metrics table."""
    rows: list[list[str]] = [
        ["Edge (%)", f"{bbr.edge_pct:.2f}%"],
        ["Consensus prob", f"{bbr.consensus_prob:.4f}"],
        ["Best odds (American)", f"{bbr.best_odds_american:+.0f}"],
        ["Best odds (Decimal)", f"{bbr.best_odds_decimal:.4f}"],
        ["Best book", bbr.best_sportsbook or entry.get("best_sportsbook", "")],
        ["Quality tier", bbr.quality_tier],
        ["Quality score", str(bbr.quality_score)],
        ["Edge Z", f"{bbr.edge_z:+.2f}"],
        ["Recency weight", f"{bbr.recency_weight:.4f}"],
        ["Volatility sigma", f"{bbr.volatility_sigma:.4f}"],
        ["Outliers removed", str(bbr.outliers_removed)],
        [
            "Kelly suggested",
            f"{bbr.kelly_suggested * 100:.2f}%"
            if bbr.kelly_suggested
            else "N/A",
        ],
        ["Sizing note", bbr.sizing_note or "—"],
        ["Books used", ", ".join(bbr.books_used) if bbr.books_used else "—"],
    ]
    return rows

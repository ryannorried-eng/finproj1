"""NRFI/YRFI Predictions Streamlit page.

Shows model predictions for first-inning scoring with edge calculations.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


def _color_edge(edge: float | None) -> str:
    if edge is None:
        return "color: gray"
    if edge > 0.05:
        return "color: green; font-weight: bold"
    if edge < 0:
        return "color: red"
    return "color: gray"


def render_nrfi_page() -> None:
    """Render the NRFI/YRFI predictions Streamlit page."""
    st.title("🎯 NRFI/YRFI Predictions")
    st.caption(
        "Predicts whether at least one run will score in the first inning "
        "(YRFI) or no runs will score (NRFI)."
    )

    # Date picker + refresh
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_date = st.date_input(
            "Date",
            value=date.today(),
            min_value=date.today() - timedelta(days=7),
            max_value=date.today() + timedelta(days=3),
        )
    with col2:
        refresh = st.button("🔄 Refresh", use_container_width=True)

    # Cache key
    cache_key = f"nrfi_preds_{selected_date}"
    if refresh and cache_key in st.session_state:
        del st.session_state[cache_key]

    # Load predictions
    if cache_key not in st.session_state:
        with st.spinner("Loading NRFI/YRFI predictions..."):
            try:
                from line_tracker.model.mlb_nrfi import predict_nrfi
                preds = predict_nrfi(game_date=selected_date)
                st.session_state[cache_key] = preds
            except FileNotFoundError:
                st.warning(
                    "NRFI model not trained yet. "
                    "Run `python -m line_tracker train-nrfi` first."
                )
                return
            except Exception as exc:
                log.exception("NRFI prediction error")
                st.error(f"Prediction error: {exc}")
                return

    preds = st.session_state.get(cache_key, [])

    if not preds:
        st.info(f"No NRFI/YRFI predictions available for {selected_date}.")
        return

    # Build DataFrame for display
    rows = []
    for p in preds:
        edge = p.get("edge")
        bet  = p.get("bet") or "—"
        rows.append({
            "Game":           p["game"],
            "Home SP":        p.get("home_sp", "TBD"),
            "Away SP":        p.get("away_sp", "TBD"),
            "YRFI%":          f"{p['yrfi_prob']:.1%}",
            "Bet":            bet,
            "Edge":           f"{edge*100:+.1f}%" if edge else "—",
            "Confidence":     p.get("confidence") or "—",
            # Hidden numeric for sorting
            "_edge_num":      edge or 0.0,
            "_yrfi_prob_num": p["yrfi_prob"],
        })

    df = pd.DataFrame(rows).sort_values("_edge_num", ascending=False)
    display_df = df.drop(columns=["_edge_num", "_yrfi_prob_num"])

    # Apply edge coloring
    def _style_row(row):
        styles = [""] * len(row)
        edge_col_idx = list(display_df.columns).index("Edge")
        raw = df.loc[row.name, "_edge_num"]
        if raw > 0.05:
            styles[edge_col_idx] = "color: green; font-weight: bold"
        elif raw < 0:
            styles[edge_col_idx] = "color: red"
        else:
            styles[edge_col_idx] = "color: gray"
        return styles

    st.dataframe(
        display_df.style.apply(_style_row, axis=1),
        use_container_width=True,
        height=min(400, 40 + 35 * len(display_df)),
    )

    # Expandable detail per game
    st.subheader("Game Details")
    for p in preds:
        with st.expander(p["game"]):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("YRFI%",  f"{p['yrfi_prob']:.1%}")
            c2.metric("NRFI%",  f"{p['nrfi_prob']:.1%}")
            edge = p.get("edge")
            c3.metric("Edge",   f"{edge*100:+.1f}%" if edge else "—")
            c4.metric("Bet",    p.get("bet") or "No edge")

            st.markdown("**Pitcher stats**")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.write(f"**Home SP:** {p.get('home_sp','TBD')}")
                st.write(f"FI YRFI rate: {p.get('home_sp_fi_yrfi_rate', 0):.1%}")
                st.write(f"Top-3 OBP: {p.get('home_top3_obp', 0):.3f}")
            with pc2:
                st.write(f"**Away SP:** {p.get('away_sp','TBD')}")
                st.write(f"FI YRFI rate: {p.get('away_sp_fi_yrfi_rate', 0):.1%}")
                st.write(f"Top-3 OBP: {p.get('away_top3_obp', 0):.3f}")

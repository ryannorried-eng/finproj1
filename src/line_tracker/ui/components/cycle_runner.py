"""Streamlit component: Run Cycle Now + latest cycle summary.

Renders an admin panel inside the sidebar Diagnostics expander that
lets users trigger a full automation cycle and view the result.
"""

from __future__ import annotations

import streamlit as st


def render_cycle_panel(store, sport_key: str, api_key: str | None) -> None:
    """Render the 'Run Cycle Now' button and latest result summary.

    Parameters
    ----------
    store : LineStore
        Open database handle (must remain open for the duration).
    sport_key : str
        The Odds API sport key (e.g. ``"basketball_nba"``).
    api_key : str | None
        Odds API key.  Passed to ``run_full_cycle`` for optional
        ingestion.  If ``None``, ingestion is skipped.
    """
    st.markdown("##### Automation Cycle")

    if st.button("Run Cycle Now", key="btn_run_cycle"):
        _execute_cycle(store, sport_key, api_key)

    # Always show the latest result if one exists in session state
    result = st.session_state.get("last_cycle_result")
    if result is not None:
        _render_result(result)


def _execute_cycle(
    store, sport_key: str, api_key: str | None
) -> None:
    """Run the full cycle and stash the result in session state."""
    from line_tracker.services.automation_service import run_full_cycle

    with st.spinner("Running automation cycle..."):
        result = run_full_cycle(
            store,
            sport=sport_key,
            api_key=api_key,
        )

    st.session_state["last_cycle_result"] = result


def _render_result(result: dict) -> None:
    """Display the structured cycle result."""
    has_errors = bool(result.get("errors"))

    # Status badge
    if has_errors:
        st.error("Last cycle completed with errors")
    else:
        st.success("Last cycle completed successfully")

    # Compact metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Lines fetched", result.get("lines_fetched", 0))
    c2.metric("Events", result.get("events_processed", 0))
    c3.metric("Picks", result.get("picks_generated", 0))

    c4, c5 = st.columns(2)
    c4.metric("Snapshots", result.get("snapshots_written", 0))
    c5.metric(
        "CLV closed",
        f"{result.get('clv_updates_completed', 0)}"
        f"/{result.get('clv_updates_attempted', 0)}",
    )

    # Timing
    started = result.get("started_at", "")
    finished = result.get("finished_at", "")
    if started and finished:
        st.caption(f"Started: {started}  \nFinished: {finished}")

    # Warnings / errors
    for w in result.get("warnings", []):
        st.warning(w, icon="\u26a0\ufe0f")
    for e in result.get("errors", []):
        st.error(e)

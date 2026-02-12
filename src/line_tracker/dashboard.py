"""Streamlit web dashboard — run with: streamlit run src/line_tracker/dashboard.py"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from line_tracker.alerts import AlertManager
from line_tracker.arbitrage import find_moneyline_arbs, find_spread_arbs
from line_tracker.models import BetType
from line_tracker.movements import detect_moves
from line_tracker.scraper import OddsClient
from line_tracker.storage import LineStore

SPORTS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
    "NCAAF": "americanfootball_ncaaf",
    "NCAAB": "basketball_ncaab",
    "MMA": "mma_mixed_martial_arts",
    "Soccer (EPL)": "soccer_epl",
}

DB_PATH = "lines.db"


def main():
    st.set_page_config(
        page_title="Line Tracker",
        page_icon="$",
        layout="wide",
    )
    st.title("Sports Betting Line Tracker")

    # --- Sidebar: API key + sport selection ---
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Odds API Key",
            value=os.environ.get("ODDS_API_KEY", ""),
            type="password",
            help="Get a free key at https://the-odds-api.com",
        )
        sport_name = st.selectbox("Sport", list(SPORTS.keys()))
        sport_key = SPORTS[sport_name]

        st.divider()
        st.caption(
            "Free tier: 500 requests/month. "
            "Each fetch uses 1-3 requests."
        )

    # --- Tabs ---
    tab_live, tab_arbs, tab_moves, tab_history = st.tabs(
        ["Live Odds", "Arbitrage Scanner", "Line Movements", "History"]
    )

    # ========== TAB 1: Live Odds ==========
    with tab_live:
        st.subheader(f"Live {sport_name} Odds")

        if st.button("Fetch Latest Odds", type="primary"):
            if not api_key:
                st.error("Enter your API key in the sidebar.")
            else:
                _fetch_and_display(api_key, sport_key, sport_name)

    # ========== TAB 2: Arbitrage ==========
    with tab_arbs:
        st.subheader(f"{sport_name} Arbitrage Scanner")

        if st.button("Scan for Arbitrage", type="primary"):
            if not api_key:
                st.error("Enter your API key in the sidebar.")
            else:
                _scan_arbs(api_key, sport_key)

    # ========== TAB 3: Line Movements ==========
    with tab_moves:
        st.subheader("Line Movements")
        threshold = st.slider(
            "Minimum change to show",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
        )
        if st.button("Detect Movements"):
            _show_movements(threshold)

    # ========== TAB 4: History ==========
    with tab_history:
        st.subheader("Stored Lines")
        _show_history()


def _fetch_and_display(api_key: str, sport_key: str, sport_name: str):
    with st.spinner("Fetching odds..."):
        try:
            with OddsClient(api_key=api_key) as client:
                lines = client.get_odds(sport=sport_key)
        except Exception as e:
            st.error(f"API error: {e}")
            return

    if not lines:
        st.warning("No odds available right now.")
        return

    # Save to DB
    with LineStore(DB_PATH) as store:
        count = store.save_lines(lines)
    st.success(f"Fetched {len(lines)} lines from {sport_name}. "
               f"Saved {count} to database.")

    # Display as table
    df = _lines_to_df(lines)

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        events = ["All"] + sorted(df["Event"].unique().tolist())
        event_filter = st.selectbox("Filter by event", events,
                                    key="live_event")
    with col2:
        types = ["All", "moneyline", "spread", "total"]
        type_filter = st.selectbox("Filter by type", types,
                                   key="live_type")

    if event_filter != "All":
        df = df[df["Event"] == event_filter]
    if type_filter != "All":
        df = df[df["Type"] == type_filter]

    st.dataframe(df, use_container_width=True, hide_index=True)


def _scan_arbs(api_key: str, sport_key: str):
    with st.spinner("Scanning for arbitrage..."):
        try:
            with OddsClient(api_key=api_key) as client:
                lines = client.get_odds(sport=sport_key)
        except Exception as e:
            st.error(f"API error: {e}")
            return

    ml_arbs = find_moneyline_arbs(lines)
    spread_arbs = find_spread_arbs(lines)
    all_arbs = ml_arbs + spread_arbs

    if not all_arbs:
        st.info("No arbitrage opportunities found.")
        return

    profitable = [a for a in all_arbs if a.profitable]
    near = [a for a in all_arbs if not a.profitable]

    if profitable:
        st.success(f"Found {len(profitable)} profitable arb(s)!")
        for arb in profitable:
            with st.container(border=True):
                st.markdown(f"**{arb.event}** — {arb.bet_type.value}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Book A", arb.side_a.sportsbook)
                c2.metric("Book B", arb.side_b.sportsbook)
                c3.metric("Margin", f"{arb.margin:+.2f}%")

    if near:
        with st.expander(f"Near-arbs ({len(near)})"):
            for arb in near:
                st.write(
                    f"{arb.event} | {arb.bet_type.value} | "
                    f"{arb.side_a.sportsbook} vs "
                    f"{arb.side_b.sportsbook} | "
                    f"{arb.margin:+.2f}%"
                )


def _show_movements(threshold: float):
    with LineStore(DB_PATH) as store:
        all_lines = store.get_lines(limit=10000)

    if len(all_lines) < 2:
        st.warning(
            "Need at least 2 snapshots. "
            "Fetch odds multiple times first."
        )
        return

    sorted_lines = sorted(all_lines, key=lambda ln: ln.timestamp)
    mid = len(sorted_lines) // 2
    old_snap = sorted_lines[:mid]
    new_snap = sorted_lines[mid:]

    moves = detect_moves(old_snap, new_snap, threshold=threshold)

    if not moves:
        st.info("No line movements detected.")
        return

    mgr = AlertManager()
    alerts = mgr.check_movements(moves)

    # Show alerts first
    for alert in alerts:
        if alert.level.value == "critical":
            st.error(alert.message)
        elif alert.level.value == "warning":
            st.warning(alert.message)
        else:
            st.info(alert.message)

    # Show moves table
    rows = []
    for m in moves:
        rows.append({
            "Book": m.sportsbook,
            "Event": m.event,
            "Type": m.bet_type.value,
            "Old": m.old_line.home_value,
            "New": m.new_line.home_value,
            "Change": m.change,
            "Direction": m.direction,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _show_history():
    with LineStore(DB_PATH) as store:
        events = store.get_events()

    if not events:
        st.warning("No data yet. Fetch some odds first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        event_filter = st.selectbox(
            "Event", ["All"] + events, key="hist_event"
        )
    with col2:
        type_filter = st.selectbox(
            "Bet type",
            ["All", "moneyline", "spread", "total"],
            key="hist_type",
        )

    bt = None
    ev = None
    if event_filter != "All":
        ev = event_filter
    if type_filter != "All":
        bt = BetType(type_filter)

    with LineStore(DB_PATH) as store:
        lines = store.get_lines(event=ev, bet_type=bt, limit=200)

    if not lines:
        st.info("No matching lines.")
        return

    df = _lines_to_df(lines)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(lines)} lines")


def _lines_to_df(lines) -> pd.DataFrame:
    rows = []
    for ln in lines:
        row = {
            "Book": ln.sportsbook,
            "Event": ln.event,
            "Type": ln.bet_type.value,
            "Home": ln.home_value,
            "Away": ln.away_value,
            "Time": ln.timestamp.strftime("%m/%d %I:%M%p"),
        }
        if ln.home_price is not None:
            row["H.Price"] = ln.home_price
            row["A.Price"] = ln.away_price
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

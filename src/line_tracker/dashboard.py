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

BET_TYPE_LABELS = {
    "moneyline": "Moneyline (who wins)",
    "spread": "Spread (point handicap)",
    "total": "Total (over/under)",
}

DB_PATH = "lines.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_key() -> str | None:
    """Return the API key or show an error. Returns None when missing."""
    key = st.session_state.get("api_key", "")
    if not key:
        st.error(
            "Paste your API key in the sidebar first.  \n"
            "Get a free one at https://the-odds-api.com"
        )
        return None
    return key


def _lines_to_df(lines) -> pd.DataFrame:
    rows = []
    for ln in lines:
        row = {
            "Sportsbook": ln.sportsbook,
            "Game": ln.event,
            "Bet Type": BET_TYPE_LABELS.get(ln.bet_type.value, ln.bet_type.value),
            "Home": ln.home_value,
            "Away": ln.away_value,
            "Fetched": ln.timestamp.strftime("%b %d, %I:%M %p"),
        }
        if ln.home_price is not None:
            row["Home Juice"] = ln.home_price
            row["Away Juice"] = ln.away_price
        rows.append(row)
    return pd.DataFrame(rows)


def _sport_name() -> str:
    return st.session_state.get("sport_name", "NFL")


def _sport_key() -> str:
    return SPORTS[_sport_name()]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar():
    with st.sidebar:
        st.header("Settings")

        st.text_input(
            "API Key",
            value=os.environ.get("ODDS_API_KEY", ""),
            type="password",
            key="api_key",
            help=(
                "Paste your key from https://the-odds-api.com.  \n"
                "The free plan gives you 500 requests/month."
            ),
        )

        st.selectbox(
            "Sport",
            list(SPORTS.keys()),
            key="sport_name",
            help="Pick the league you want to track.",
        )

        st.divider()

        st.subheader("Quick glossary")
        st.caption(
            "**Moneyline** — Bet on who wins.  \n"
            "**Spread** — Team must win (or lose) by "
            "a certain number of points.  \n"
            "**Total (O/U)** — Bet on whether the combined "
            "score is over or under a number.  \n"
            "**Arbitrage** — When different sportsbooks "
            "disagree enough that you can bet both sides "
            "and guarantee a profit."
        )

        st.divider()
        st.caption(
            "Free tier: 500 requests/month.  \n"
            "Each \"Fetch\" uses 1-3 requests."
        )


# ---------------------------------------------------------------------------
# Tab: Live Odds
# ---------------------------------------------------------------------------

def _tab_live():
    sport = _sport_name()
    st.subheader(f"Live {sport} Odds")
    st.caption(
        "Click the button to pull the latest lines from every major sportsbook."
    )

    if st.button("Fetch Latest Odds", type="primary", key="btn_fetch"):
        api_key = _require_key()
        if api_key is None:
            return
        with st.spinner("Pulling odds from sportsbooks..."):
            try:
                with OddsClient(api_key=api_key) as client:
                    lines = client.get_odds(sport=_sport_key())
            except Exception as exc:
                st.error(f"Could not reach the API: {exc}")
                return

        if not lines:
            st.info(f"No {sport} games available right now.")
            return

        # Persist in DB
        with LineStore(DB_PATH) as store:
            count = store.save_lines(lines)

        st.success(f"Got {len(lines)} lines across {sport}. Saved {count} new rows.")

        # Store in session for filtering without re-fetching
        st.session_state["last_fetch"] = lines

    lines = st.session_state.get("last_fetch")
    if not lines:
        return

    df = _lines_to_df(lines)

    # Summary metrics
    books = df["Sportsbook"].nunique()
    games = df["Game"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Sportsbooks", books)
    c2.metric("Games", games)
    c3.metric("Total Lines", len(df))

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        events = ["All games"] + sorted(df["Game"].unique().tolist())
        event_pick = st.selectbox("Game", events, key="live_game")
    with col2:
        type_opts = ["All types"] + list(BET_TYPE_LABELS.values())
        type_pick = st.selectbox("Bet type", type_opts, key="live_type")
    with col3:
        book_opts = ["All books"] + sorted(df["Sportsbook"].unique().tolist())
        book_pick = st.selectbox("Sportsbook", book_opts, key="live_book")

    filtered = df.copy()
    if event_pick != "All games":
        filtered = filtered[filtered["Game"] == event_pick]
    if type_pick != "All types":
        filtered = filtered[filtered["Bet Type"] == type_pick]
    if book_pick != "All books":
        filtered = filtered[filtered["Sportsbook"] == book_pick]

    st.dataframe(filtered, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Best Lines
# ---------------------------------------------------------------------------

def _tab_best_lines():
    st.subheader("Best Available Lines")
    st.caption(
        "Shows the best odds each sportsbook is offering right now "
        "for every game. Fetches from your saved data — "
        "go to **Live Odds** and fetch first if this is empty."
    )

    with LineStore(DB_PATH) as store:
        events = store.get_events()

    if not events:
        st.info("No data yet. Head to the **Live Odds** tab and fetch some odds first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        event_pick = st.selectbox("Game", events, key="best_event")
    with col2:
        bt_pick = st.selectbox(
            "Bet type",
            list(BET_TYPE_LABELS.keys()),
            format_func=lambda k: BET_TYPE_LABELS[k],
            key="best_type",
        )

    bt = BetType(bt_pick)
    with LineStore(DB_PATH) as store:
        latest = store.get_latest_for_event(event_pick, bt)

    if not latest:
        st.info("No lines found for this game + bet type combination.")
        return

    df = _lines_to_df(latest)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Highlight the best values
    if bt == BetType.MONEYLINE:
        best_home = max(latest, key=lambda ln: ln.home_value)
        best_away = max(latest, key=lambda ln: ln.away_value)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                f"Best Home odds: {best_home.sportsbook}",
                _format_odds(best_home.home_value),
            )
        with c2:
            st.metric(
                f"Best Away odds: {best_away.sportsbook}",
                _format_odds(best_away.away_value),
            )
    elif bt == BetType.SPREAD:
        best_home = max(latest, key=lambda ln: ln.home_value)
        best_away = max(latest, key=lambda ln: ln.away_value)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                f"Best Home spread: {best_home.sportsbook}",
                f"{best_home.home_value:+.1f}",
            )
        with c2:
            st.metric(
                f"Best Away spread: {best_away.sportsbook}",
                f"{best_away.away_value:+.1f}",
            )


def _format_odds(val: float) -> str:
    """Format American odds with a + or - sign."""
    return f"{val:+.0f}" if val != 0 else "EVEN"


# ---------------------------------------------------------------------------
# Tab: Arbitrage
# ---------------------------------------------------------------------------

def _tab_arbs():
    sport = _sport_name()
    st.subheader(f"{sport} Arbitrage Scanner")
    st.caption(
        "Scans every sportsbook for pricing disagreements.  \n"
        "A **profitable arb** means you can bet both sides and guarantee a profit. "
        "A **near-arb** is close but not quite there yet — worth watching."
    )

    if st.button("Scan for Arbitrage", type="primary", key="btn_arb"):
        api_key = _require_key()
        if api_key is None:
            return

        with st.spinner("Comparing odds across all sportsbooks..."):
            try:
                with OddsClient(api_key=api_key) as client:
                    lines = client.get_odds(sport=_sport_key())
            except Exception as exc:
                st.error(f"Could not reach the API: {exc}")
                return

        ml_arbs = find_moneyline_arbs(lines)
        spread_arbs = find_spread_arbs(lines)
        all_arbs = ml_arbs + spread_arbs

        # Use AlertManager to classify
        mgr = AlertManager(arb_min_margin=-5.0)
        alerts = mgr.check_arbitrage(all_arbs)

        st.session_state["last_arbs"] = all_arbs
        st.session_state["last_arb_alerts"] = alerts

    all_arbs = st.session_state.get("last_arbs")
    alerts = st.session_state.get("last_arb_alerts", [])

    if all_arbs is None:
        return

    if not all_arbs:
        st.info(
            "No arbitrage opportunities right now.  \n"
            "This is normal — true arbs are rare and short-lived."
        )
        return

    profitable = [a for a in all_arbs if a.profitable]
    near = [a for a in all_arbs if not a.profitable]

    # Summary
    c1, c2 = st.columns(2)
    c1.metric("Profitable Arbs", len(profitable))
    c2.metric("Near Arbs", len(near))

    # Show relevant alerts
    for alert in alerts:
        if alert.level.value == "critical":
            st.error(alert.message)

    # Profitable arbs
    if profitable:
        st.success(f"Found {len(profitable)} profitable arb(s)!")
        for arb in profitable:
            with st.container(border=True):
                st.markdown(
                    f"**{arb.event}** — "
                    f"{BET_TYPE_LABELS.get(arb.bet_type.value, arb.bet_type.value)}"
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Bet side A at",
                    arb.side_a.sportsbook,
                    help=f"Home value: {arb.side_a.home_value}",
                )
                c2.metric(
                    "Bet side B at",
                    arb.side_b.sportsbook,
                    help=f"Away value: {arb.side_b.away_value}",
                )
                c3.metric("Profit margin", f"{arb.margin:+.2f}%")

    # Near arbs
    if near:
        with st.expander(f"Near-arbs ({len(near)}) — not profitable yet, but close"):
            rows = []
            for arb in near:
                rows.append({
                    "Game": arb.event,
                    "Type": BET_TYPE_LABELS.get(arb.bet_type.value, arb.bet_type.value),
                    "Book A": arb.side_a.sportsbook,
                    "Book B": arb.side_b.sportsbook,
                    "Margin": f"{arb.margin:+.2f}%",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Tab: Line Movements
# ---------------------------------------------------------------------------

def _tab_movements():
    st.subheader("Line Movements")
    st.caption(
        "Compares your older fetches with newer ones to spot which lines moved. "
        "Big moves can signal sharp action or breaking news."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        threshold = st.slider(
            "Only show changes bigger than",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            help=(
                "For moneyline, this is the odds change "
                "(e.g. 10 = a shift of 10 points).  \n"
                "For spreads/totals, this is the point "
                "change (e.g. 0.5 = half a point)."
            ),
        )
    with col2:
        st.write("")  # spacer
        st.write("")
        detect = st.button("Detect Movements", key="btn_moves")

    if detect:
        with LineStore(DB_PATH) as store:
            all_lines = store.get_lines(limit=10000)

        if len(all_lines) < 2:
            st.warning(
                "You need at least **two snapshots** to compare.  \n"
                "Go to **Live Odds**, fetch now, wait a while, then fetch again."
            )
            return

        sorted_lines = sorted(all_lines, key=lambda ln: ln.timestamp)
        mid = len(sorted_lines) // 2
        old_snap = sorted_lines[:mid]
        new_snap = sorted_lines[mid:]

        moves = detect_moves(old_snap, new_snap, threshold=threshold)
        mgr = AlertManager()
        alerts = mgr.check_movements(moves)

        st.session_state["last_moves"] = moves
        st.session_state["last_move_alerts"] = alerts

    moves = st.session_state.get("last_moves")
    alerts = st.session_state.get("last_move_alerts", [])

    if moves is None:
        return

    if not moves:
        st.info("No line movements detected with the current threshold.")
        return

    # Show alerts
    for alert in alerts:
        if alert.level.value == "critical":
            st.error(alert.message)
        elif alert.level.value == "warning":
            st.warning(alert.message)

    # Summary
    up = sum(1 for m in moves if m.direction == "up")
    down = sum(1 for m in moves if m.direction == "down")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Moves", len(moves))
    c2.metric("Moved Up", up)
    c3.metric("Moved Down", down)

    # Table
    rows = []
    for m in moves:
        rows.append({
            "Sportsbook": m.sportsbook,
            "Game": m.event,
            "Bet Type": BET_TYPE_LABELS.get(m.bet_type.value, m.bet_type.value),
            "Was": m.old_line.home_value,
            "Now": m.new_line.home_value,
            "Change": f"{m.change:+.1f}",
            "Direction": m.direction,
        })
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True
    )


# ---------------------------------------------------------------------------
# Tab: History
# ---------------------------------------------------------------------------

def _tab_history():
    st.subheader("Saved History")
    st.caption(
        "Browse every line you've fetched. "
        "Use the filters to narrow it down."
    )

    with LineStore(DB_PATH) as store:
        events = store.get_events()

    if not events:
        st.info(
            "Nothing saved yet.  \n"
            "Go to the **Live Odds** tab and fetch some odds to get started."
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        event_pick = st.selectbox(
            "Game", ["All games"] + events, key="hist_event"
        )
    with col2:
        type_pick = st.selectbox(
            "Bet type",
            ["All types"] + list(BET_TYPE_LABELS.keys()),
            format_func=lambda k: BET_TYPE_LABELS.get(k, k),
            key="hist_type",
        )
    with col3:
        limit = st.select_slider(
            "Max rows",
            options=[50, 100, 200, 500],
            value=100,
            key="hist_limit",
        )

    ev = None if event_pick == "All games" else event_pick
    bt = None if type_pick == "All types" else BetType(type_pick)

    with LineStore(DB_PATH) as store:
        lines = store.get_lines(event=ev, bet_type=bt, limit=limit)

    if not lines:
        st.info("No matching lines for these filters.")
        return

    df = _lines_to_df(lines)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(lines)} of up to {limit} rows")


# ---------------------------------------------------------------------------
# Tab: Available Sports
# ---------------------------------------------------------------------------

def _tab_sports():
    st.subheader("Available Sports")
    st.caption(
        "These are all the sports currently available from the API. "
        "This is a free call — it does not count against your quota."
    )

    if st.button("Load Sports List", key="btn_sports"):
        api_key = _require_key()
        if api_key is None:
            return
        with st.spinner("Loading..."):
            try:
                with OddsClient(api_key=api_key) as client:
                    sports = client.get_sports()
            except Exception as exc:
                st.error(f"Could not reach the API: {exc}")
                return
        st.session_state["sports_list"] = sports

    sports = st.session_state.get("sports_list")
    if sports is None:
        return

    active = [s for s in sports if s.get("active")]
    inactive = [s for s in sports if not s.get("active")]

    st.metric("Active Sports", len(active))

    rows = []
    for s in active:
        rows.append({
            "Sport": s.get("title", ""),
            "API Key": s.get("key", ""),
            "Group": s.get("group", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if inactive:
        with st.expander(f"Off-season / inactive ({len(inactive)})"):
            rows = []
            for s in inactive:
                rows.append({
                    "Sport": s.get("title", ""),
                    "API Key": s.get("key", ""),
                    "Group": s.get("group", ""),
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Line Tracker",
        page_icon="$",
        layout="wide",
    )

    st.title("Sports Betting Line Tracker")
    st.caption(
        "Compare odds across sportsbooks, find arbitrage, "
        "and track how lines move over time."
    )

    _sidebar()

    tabs = st.tabs([
        "Live Odds",
        "Best Lines",
        "Arbitrage",
        "Line Movements",
        "History",
        "Sports List",
    ])

    with tabs[0]:
        _tab_live()
    with tabs[1]:
        _tab_best_lines()
    with tabs[2]:
        _tab_arbs()
    with tabs[3]:
        _tab_movements()
    with tabs[4]:
        _tab_history()
    with tabs[5]:
        _tab_sports()


if __name__ == "__main__":
    main()

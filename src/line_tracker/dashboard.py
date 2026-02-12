"""Streamlit web dashboard — run with: streamlit run src/line_tracker/dashboard.py"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

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

BET_TYPE_SHORT = {
    "moneyline": "ML",
    "spread": "Spread",
    "total": "O/U",
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


def _format_odds(val: float) -> str:
    """Format American odds with a + or - sign."""
    return f"{val:+.0f}" if val != 0 else "EVEN"


def _format_value(val: float, bet_type: BetType) -> str:
    """Format a line value based on bet type."""
    if bet_type == BetType.MONEYLINE:
        return _format_odds(val)
    if bet_type == BetType.SPREAD:
        return f"{val:+.1f}"
    if bet_type == BetType.TOTAL:
        return f"{val:.1f}"
    return str(val)


def _relative_time(dt: datetime) -> str:
    """Format a datetime as relative time like '3m ago'."""
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    seconds = (now - dt).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 0:
        return american / 100 + 1
    return 100 / abs(american) + 1


def _max_display_rows() -> int:
    return st.session_state.get("max_rows", 200)


def _compact_mode() -> bool:
    return st.session_state.get("compact_mode", False)


def _lines_to_df(lines, compact: bool = False) -> pd.DataFrame:
    """Convert BettingLine objects to a display DataFrame with formatted values."""
    labels = BET_TYPE_SHORT if compact else BET_TYPE_LABELS
    rows = []
    for ln in lines:
        row = {
            "Sportsbook": ln.sportsbook,
            "Game": ln.event,
            "Bet Type": labels.get(ln.bet_type.value, ln.bet_type.value),
            "Home": _format_value(ln.home_value, ln.bet_type),
            "Away": _format_value(ln.away_value, ln.bet_type),
            "Fetched": _relative_time(ln.timestamp),
        }
        if not compact and ln.home_price is not None:
            row["Home Juice"] = _format_odds(ln.home_price)
            row["Away Juice"] = _format_odds(ln.away_price)
        rows.append(row)
    return pd.DataFrame(rows)


def _lines_to_raw_df(lines) -> pd.DataFrame:
    """Convert BettingLine objects to a raw numeric DataFrame."""
    rows = []
    for ln in lines:
        rows.append({
            "Sportsbook": ln.sportsbook,
            "Game": ln.event,
            "Bet Type": ln.bet_type.value,
            "Home": ln.home_value,
            "Away": ln.away_value,
        })
    return pd.DataFrame(rows)


def _sport_name() -> str:
    return st.session_state.get("sport_name", "NFL")


def _sport_key() -> str:
    return SPORTS[_sport_name()]


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_BEST_CELL = "background-color: #d4edda; font-weight: bold"


def _style_best_lines(
    display_df: pd.DataFrame, raw_df: pd.DataFrame,
) -> pd.io.formats.style.Styler:
    """Highlight best Home/Away values per Game+BetType group."""
    css = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
    if raw_df.empty:
        return display_df.style

    for _, grp in raw_df.groupby(["Game", "Bet Type"]):
        if len(grp) < 2:
            continue
        if "Home" in css.columns:
            css.loc[grp["Home"].idxmax(), "Home"] = _BEST_CELL
        if "Away" in css.columns:
            css.loc[grp["Away"].idxmax(), "Away"] = _BEST_CELL

    return display_df.style.apply(lambda _: css, axis=None)


def _highlight_best(
    display_df: pd.DataFrame, raw_df: pd.DataFrame,
) -> pd.io.formats.style.Styler:
    """Highlight the best Home/Away values in a single-group table."""
    css = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
    if raw_df.empty or len(raw_df) < 2:
        return display_df.style

    if "Home" in css.columns and "Home" in raw_df.columns:
        css.loc[raw_df["Home"].idxmax(), "Home"] = _BEST_CELL
    if "Away" in css.columns and "Away" in raw_df.columns:
        css.loc[raw_df["Away"].idxmax(), "Away"] = _BEST_CELL

    return display_df.style.apply(lambda _: css, axis=None)


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

        st.subheader("Display")
        st.select_slider(
            "Max rows to show",
            options=[50, 100, 200, 500, 1000],
            value=200,
            key="max_rows",
            help="Limit the number of rows displayed in tables.",
        )
        st.toggle(
            "Compact mode",
            value=False,
            key="compact_mode",
            help=(
                "Hides juice columns and uses shorter bet-type labels "
                "for a denser view."
            ),
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
# Arb stake calculator
# ---------------------------------------------------------------------------

def _compute_arb_stakes(
    odds_a: float, odds_b: float, total_stake: float = 100.0,
) -> tuple[float, float, float, float] | None:
    """Compute optimal stake split for a two-leg moneyline arb.

    Returns (stake_a, stake_b, guaranteed_profit, roi_pct) or None.
    """
    dec_a = _american_to_decimal(odds_a)
    dec_b = _american_to_decimal(odds_b)
    if dec_a <= 1 or dec_b <= 1:
        return None

    inv_a = 1 / dec_a
    inv_b = 1 / dec_b
    total_implied = inv_a + inv_b
    if total_implied <= 0:
        return None

    stake_a = total_stake * inv_a / total_implied
    stake_b = total_stake * inv_b / total_implied
    payout = stake_a * dec_a  # same as stake_b * dec_b for a perfect split
    profit = payout - total_stake
    roi = (profit / total_stake) * 100

    return round(stake_a, 2), round(stake_b, 2), round(profit, 2), round(roi, 2)


# ---------------------------------------------------------------------------
# Data fetching and indexing
# ---------------------------------------------------------------------------

def _fetch_odds():
    """Fetch latest odds, save to DB, and auto-scan for arbs/movements."""
    sport = _sport_name()
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

    with LineStore(DB_PATH) as store:
        count = store.save_lines(lines)

    st.success(f"Got {len(lines)} lines across {sport}. Saved {count} new rows.")
    st.session_state["last_fetch"] = lines

    # Auto-scan for arbs
    ml_arbs = find_moneyline_arbs(lines)
    spread_arbs = find_spread_arbs(lines)
    st.session_state["last_arbs"] = ml_arbs + spread_arbs

    # Auto-detect movements from stored history
    with LineStore(DB_PATH) as store:
        all_lines = store.get_lines(limit=10000)
    if len(all_lines) >= 2:
        sorted_lines = sorted(all_lines, key=lambda ln: ln.timestamp)
        mid = len(sorted_lines) // 2
        moves = detect_moves(sorted_lines[:mid], sorted_lines[mid:])
        st.session_state["last_moves"] = moves


def _build_game_index(lines) -> dict[str, dict]:
    """Group fetched lines into a dict keyed by event name."""
    games: dict[str, dict] = {}
    for ln in lines:
        if ln.event not in games:
            games[ln.event] = {
                "books": set(),
                "last_updated": ln.timestamp,
                "home_team": ln.home_team,
                "away_team": ln.away_team,
                "sport": ln.sport,
                "bet_types": set(),
            }
        g = games[ln.event]
        g["books"].add(ln.sportsbook)
        g["bet_types"].add(ln.bet_type)
        if ln.timestamp > g["last_updated"]:
            g["last_updated"] = ln.timestamp
    return games


# ---------------------------------------------------------------------------
# PAGE 1: Dashboard — game list
# ---------------------------------------------------------------------------

def _page_dashboard():
    sport = _sport_name()
    st.title("Sports Betting Line Tracker")
    st.caption(
        "Compare odds across sportsbooks, find arbitrage, "
        "and track how lines move over time."
    )

    if st.button(f"Fetch Latest {sport} Odds", type="primary", key="btn_fetch"):
        _fetch_odds()

    lines = st.session_state.get("last_fetch")
    if not lines:
        st.info(
            f"No data loaded yet. Click above to fetch the latest {sport} odds."
        )
        return

    games = _build_game_index(lines)

    # Badge lookup
    all_arbs = st.session_state.get("last_arbs", [])
    arb_events = {a.event for a in all_arbs if a.profitable}
    near_arb_events = {
        a.event for a in all_arbs if not a.profitable and a.margin > -2
    }
    all_moves = st.session_state.get("last_moves", [])
    move_events = {m.event for m in all_moves} if all_moves else set()

    # Summary metrics
    all_books = {b for g in games.values() for b in g["books"]}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", len(games))
    c2.metric("Sportsbooks", len(all_books))
    c3.metric("Total Lines", len(lines))
    c4.metric("Arb Alerts", len(arb_events))

    st.divider()

    # Sort by last_updated descending
    sorted_games = sorted(
        games.items(), key=lambda x: x[1]["last_updated"], reverse=True,
    )

    for event_name, info in sorted_games:
        with st.container(border=True):
            cols = st.columns([5, 2, 2, 1])
            with cols[0]:
                st.markdown(f"**{event_name}**")
                st.caption(sport)
            with cols[1]:
                st.markdown(f"{len(info['books'])} books")
                st.caption(f"Updated {_relative_time(info['last_updated'])}")
            with cols[2]:
                badges = []
                if event_name in arb_events:
                    badges.append(":red[**ARB**]")
                elif event_name in near_arb_events:
                    badges.append(":orange[**NEAR-ARB**]")
                if event_name in move_events:
                    badges.append(":blue[**MOVE**]")
                if badges:
                    st.markdown(" \u00a0 ".join(badges))
            with cols[3]:
                if st.button("View", key=f"view_{event_name}"):
                    st.session_state["page"] = "detail"
                    st.session_state["selected_game"] = event_name
                    st.rerun()

    # Sports list (collapsed)
    with st.expander("Browse available sports"):
        _sports_list()


def _sports_list():
    """Show all available sports from the API."""
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
        st.caption("This is a free call — does not count against your quota.")
        return

    active = [s for s in sports if s.get("active")]
    st.metric("Active Sports", len(active))

    rows = []
    for s in active:
        rows.append({
            "Sport": s.get("title", ""),
            "API Key": s.get("key", ""),
            "Group": s.get("group", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 2: Game Detail
# ---------------------------------------------------------------------------

def _page_detail():
    event_name = st.session_state.get("selected_game", "")

    if st.button("\u2190 Back to Dashboard", key="btn_back"):
        st.session_state["page"] = "dashboard"
        st.rerun()

    lines = st.session_state.get("last_fetch", [])
    game_lines = [ln for ln in lines if ln.event == event_name]

    st.title(event_name)

    if game_lines:
        books = {ln.sportsbook for ln in game_lines}
        latest_ts = max(ln.timestamp for ln in game_lines)
        st.caption(
            f"{_sport_name()} \u00b7 {len(books)} sportsbooks \u00b7 "
            f"Updated {_relative_time(latest_ts)}"
        )

    tabs = st.tabs([
        "Odds Comparison", "Arbitrage", "Line Movements", "History",
    ])

    with tabs[0]:
        _detail_odds(event_name, game_lines)
    with tabs[1]:
        _detail_arbs(event_name)
    with tabs[2]:
        _detail_movements(event_name)
    with tabs[3]:
        _detail_history(event_name)


# -- Detail tab: Odds Comparison -------------------------------------------

def _detail_odds(event_name: str, game_lines):
    """Shows all sportsbook lines for this game, grouped by bet type."""
    if not game_lines:
        st.info("No odds data for this game. Fetch odds from the dashboard first.")
        return

    compact = _compact_mode()

    for bt in [BetType.MONEYLINE, BetType.SPREAD, BetType.TOTAL]:
        bt_lines = [ln for ln in game_lines if ln.bet_type == bt]
        if not bt_lines:
            continue

        label = BET_TYPE_LABELS.get(bt.value, bt.value)
        st.subheader(label)

        display_df = _lines_to_df(bt_lines, compact=compact)
        raw_df = _lines_to_raw_df(bt_lines)

        # Drop redundant Game column (all rows are the same game)
        if "Game" in display_df.columns:
            display_df = display_df.drop(columns=["Game"])

        if not display_df.empty:
            styled = _highlight_best(display_df, raw_df)
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Best-value callouts
        if bt == BetType.MONEYLINE and len(bt_lines) >= 2:
            best_home = max(bt_lines, key=lambda ln: ln.home_value)
            best_away = max(bt_lines, key=lambda ln: ln.away_value)
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    f"Best Home: {best_home.sportsbook}",
                    _format_odds(best_home.home_value),
                )
            with c2:
                st.metric(
                    f"Best Away: {best_away.sportsbook}",
                    _format_odds(best_away.away_value),
                )
        elif bt == BetType.SPREAD and len(bt_lines) >= 2:
            best_home = max(bt_lines, key=lambda ln: ln.home_value)
            best_away = max(bt_lines, key=lambda ln: ln.away_value)
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


# -- Detail tab: Arbitrage -------------------------------------------------

def _detail_arbs(event_name: str):
    """Shows arb opportunities for this game."""
    all_arbs = st.session_state.get("last_arbs", [])
    game_arbs = [a for a in all_arbs if a.event == event_name]

    if not game_arbs:
        st.info("No arbitrage opportunities detected for this game.")
        return

    profitable = [a for a in game_arbs if a.profitable]
    near = [a for a in game_arbs if not a.profitable]

    c1, c2 = st.columns(2)
    c1.metric("Profitable", len(profitable))
    c2.metric("Near Arbs", len(near))

    if profitable:
        st.success(f"Found {len(profitable)} profitable arb(s)!")
        for arb in profitable:
            with st.container(border=True):
                st.markdown(
                    f"**{BET_TYPE_LABELS.get(arb.bet_type.value, arb.bet_type.value)}**"
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Side A",
                    arb.side_a.sportsbook,
                    help=f"Home value: {arb.side_a.home_value}",
                )
                c2.metric(
                    "Side B",
                    arb.side_b.sportsbook,
                    help=f"Away value: {arb.side_b.away_value}",
                )
                c3.metric("Profit margin", f"{arb.margin:+.2f}%")

                # Stake calculator for moneyline arbs
                if arb.bet_type == BetType.MONEYLINE:
                    result = _compute_arb_stakes(
                        arb.side_a.home_value, arb.side_b.away_value,
                    )
                    if result:
                        stake_a, stake_b, profit, roi = result
                        with st.expander(
                            "Stake calculator ($100 total)", expanded=True,
                        ):
                            lc1, lc2 = st.columns(2)
                            with lc1:
                                st.markdown(
                                    f"**Leg A \u2014 {arb.side_a.sportsbook}**  \n"
                                    f"Bet Home at "
                                    f"{_format_odds(arb.side_a.home_value)}  \n"
                                    f"Stake: **${stake_a:.2f}**"
                                )
                            with lc2:
                                st.markdown(
                                    f"**Leg B \u2014 {arb.side_b.sportsbook}**  \n"
                                    f"Bet Away at "
                                    f"{_format_odds(arb.side_b.away_value)}  \n"
                                    f"Stake: **${stake_b:.2f}**"
                                )
                            st.divider()
                            pc1, pc2, pc3 = st.columns(3)
                            pc1.metric("Total wagered", "$100.00")
                            pc2.metric("Guaranteed profit", f"${profit:.2f}")
                            pc3.metric("ROI", f"{roi:.2f}%")

    if near:
        with st.expander(
            f"Near-arbs ({len(near)}) \u2014 not profitable yet, but close",
        ):
            rows = []
            for arb in near:
                rows.append({
                    "Type": BET_TYPE_LABELS.get(
                        arb.bet_type.value, arb.bet_type.value,
                    ),
                    "Book A": arb.side_a.sportsbook,
                    "Book B": arb.side_b.sportsbook,
                    "Margin": f"{arb.margin:+.2f}%",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


# -- Detail tab: Line Movements --------------------------------------------

_TIME_WINDOWS = {
    "All": 0,
    "Last 15m": 15,
    "Last 1h": 60,
    "Last 6h": 360,
    "Last 24h": 1440,
}


def _detail_movements(event_name: str):
    """Shows detected movements for this game."""
    all_moves = st.session_state.get("last_moves", [])
    game_moves = [m for m in all_moves if m.event == event_name]

    if not game_moves:
        st.info(
            "No line movements detected for this game.  \n"
            "Movements appear after multiple fetches over time."
        )
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        time_window = st.selectbox(
            "Time window",
            list(_TIME_WINDOWS.keys()),
            key="detail_move_window",
        )
    with col2:
        hide_juice = st.toggle(
            "Hide juice-only changes",
            value=False,
            key="detail_hide_juice",
        )

    # Apply filters
    filtered = list(game_moves)

    window_min = _TIME_WINDOWS.get(time_window, 0)
    if window_min > 0:
        now = datetime.now()
        kept = []
        for m in filtered:
            ts = m.new_line.timestamp
            ref = datetime.now(ts.tzinfo) if ts.tzinfo else now
            if (ref - ts).total_seconds() <= window_min * 60:
                kept.append(m)
        filtered = kept

    if hide_juice:
        kept = []
        for m in filtered:
            if m.bet_type == BetType.MONEYLINE and abs(m.change) < 1:
                continue
            if m.bet_type in (BetType.SPREAD, BetType.TOTAL) and abs(m.change) < 0.5:
                continue
            kept.append(m)
        filtered = kept

    if not filtered:
        st.info("No movements match the current filters.")
        return

    # Summary
    up = sum(1 for m in filtered if m.direction == "up")
    down = sum(1 for m in filtered if m.direction == "down")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Moves", len(filtered))
    c2.metric("Up", up)
    c3.metric("Down", down)

    # Movement table
    rows = []
    for m in filtered:
        arrow = "\u2191" if m.direction == "up" else (
            "\u2193" if m.direction == "down" else "\u2014"
        )
        old_str = _format_value(m.old_line.home_value, m.bet_type)
        new_str = _format_value(m.new_line.home_value, m.bet_type)
        rows.append({
            "Sportsbook": m.sportsbook,
            "Bet Type": BET_TYPE_LABELS.get(m.bet_type.value, m.bet_type.value),
            "Old": old_str,
            "": "\u2192",
            "New": new_str,
            "Delta": f"{arrow} {m.change:+.1f}",
            "When": _relative_time(m.new_line.timestamp),
        })

    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
    )


# -- Detail tab: History ----------------------------------------------------

def _detail_history(event_name: str):
    """Shows stored line history for this game."""
    with LineStore(DB_PATH) as store:
        lines = store.get_lines(event=event_name, limit=_max_display_rows())

    if not lines:
        st.info("No stored history for this game yet.")
        return

    compact = _compact_mode()
    df = _lines_to_df(lines, compact=compact)

    # Drop redundant Game column
    if "Game" in df.columns:
        df = df.drop(columns=["Game"])

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(lines)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Line Tracker",
        page_icon="$",
        layout="wide",
    )

    _sidebar()

    page = st.session_state.get("page", "dashboard")
    if page == "detail" and st.session_state.get("selected_game"):
        _page_detail()
    else:
        _page_dashboard()


if __name__ == "__main__":
    main()

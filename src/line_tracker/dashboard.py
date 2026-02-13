"""Streamlit web dashboard — run with: streamlit run src/line_tracker/dashboard.py"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from line_tracker.arbitrage import find_moneyline_arbs, find_spread_arbs
from line_tracker.bet_history import init_bet_state, settle_bet, submit_bet
from line_tracker.bet_slip import (
    american_profit,
    american_total_return,
    compute_standouts,
    format_american,
    has_conflicting_leg,
    is_duplicate_leg,
    parlay_payout,
)
from line_tracker.bet_slip import (
    american_to_decimal as _slip_a2d,
)
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

# Mapping from label back to bet_type value for filters
_LABEL_TO_BT = {v: k for k, v in BET_TYPE_LABELS.items()}

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


def _format_clock_time(dt: datetime) -> str:
    """Format datetime as a short clock time, e.g. '2:30 PM'."""
    local = dt.astimezone() if dt.tzinfo else dt
    return local.strftime("%I:%M %p").lstrip("0")


def _format_start_time(dt: datetime) -> str:
    """Format commence_time as 'Fri 12:15 PM'."""
    local = dt.astimezone() if dt.tzinfo else dt
    return local.strftime("%a %I:%M %p").replace(" 0", " ")


def _relative_date_label(dt: datetime) -> str:
    """Return 'Today', 'Tomorrow', or short date like 'Sat Feb 15'."""
    local = dt.astimezone() if dt.tzinfo else dt
    local_date = local.date()
    today = date.today()
    if local_date == today:
        return "Today"
    if local_date == today + timedelta(days=1):
        return "Tomorrow"
    return local_date.strftime("%a %b %-d")


def _american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    return _slip_a2d(american)


def _max_display_rows() -> int:
    return st.session_state.get("max_rows", 200)


def _compact_mode() -> bool:
    return st.session_state.get("compact_mode", False)


def fmt_money(x: float, sign: bool = False) -> str:
    """Format currency: $1,234.56.  Use *sign=True* for +$… / -$…."""
    if sign:
        prefix = "+" if x >= 0 else "-"
        return f"{prefix}${abs(x):,.2f}"
    return f"${x:,.2f}"


def fmt_pct(x: float, sign: bool = False) -> str:
    """Format percentage: 12.3%.  Use *sign=True* for +12.3% / -1.2%."""
    if sign:
        return f"{x:+.1f}%"
    return f"{x:.1f}%"


def fmt_odds(x: float) -> str:
    """Format American odds: '+120' or '-110'."""
    return format_american(x)


def _get_slip_book() -> str | None:
    """Return the currently locked sportsbook, or None."""
    return st.session_state.get("slip_book")


def _lock_slip_book(book: str) -> None:
    """Lock the bet slip to a sportsbook (called when first leg is added)."""
    st.session_state["slip_book"] = book


def _clear_slip() -> None:
    """Clear all legs and unlock sportsbook."""
    st.session_state["bet_slip"] = []
    st.session_state["slip_book"] = None
    st.session_state["slip_stake"] = 100.0


def _lines_to_shopping_entries(lines) -> list[dict]:
    """Convert BettingLine objects to entry dicts for compute_standouts.

    Produces one entry per (book, event, market, selection) from the fetched lines.
    """
    entries: list[dict] = []
    for ln in lines:
        event = ln.event
        sport = ln.sport
        book = ln.sportsbook
        if ln.bet_type == BetType.MONEYLINE:
            for side, odds in [("Home", ln.home_value), ("Away", ln.away_value)]:
                entries.append({
                    "event": event, "market": "ML", "selection": side,
                    "sportsbook": book, "odds": odds, "line": None,
                    "sport": sport,
                })
        elif ln.bet_type == BetType.SPREAD:
            for side, odds, line_val in [
                ("Home", ln.home_price, ln.home_value),
                ("Away", ln.away_price, ln.away_value),
            ]:
                if odds is None:
                    continue
                entries.append({
                    "event": event, "market": "Spread", "selection": side,
                    "sportsbook": book, "odds": odds, "line": line_val,
                    "sport": sport,
                })
        elif ln.bet_type == BetType.TOTAL:
            for side, odds, line_val in [
                ("Over", ln.home_price, ln.home_value),
                ("Under", ln.away_price, ln.away_value),
            ]:
                if odds is None:
                    continue
                entries.append({
                    "event": event, "market": "Total", "selection": side,
                    "sportsbook": book, "odds": odds, "line": line_val,
                    "sport": sport,
                })
    return entries


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
        # --- Bet Slip button (prominent, always visible) ---
        slip = st.session_state.get("bet_slip", [])
        slip_book = _get_slip_book()
        count = len(slip)
        label = f"Bet Slip ({count})" if count else "Bet Slip"

        if st.button(
            label, key="btn_open_slip", type="primary",
            use_container_width=True,
        ):
            st.session_state["_open_slip"] = True

        if count and slip_book:
            st.caption(f"Locked to {slip_book}")

        # --- Bet History button ---
        active_count = len(st.session_state.get("active_bets", []))
        settled_count = len(st.session_state.get("settled_bets", []))
        total_bets = active_count + settled_count
        hist_label = (
            f"Bet History ({total_bets})" if total_bets else "Bet History"
        )
        if st.button(
            hist_label, key="btn_open_history",
            use_container_width=True,
        ):
            st.session_state["_open_history"] = True

        st.divider()

        # --- Page navigation ---
        st.radio(
            "Page",
            ["Dashboard", "Best Lines to Shop"],
            key="nav_page",
            horizontal=True,
        )

        st.divider()

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

        with st.expander("Advanced"):
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

        with st.expander("Glossary"):
            st.caption(
                "**Moneyline** \u2014 Bet on who wins.  \n"
                "**Spread** \u2014 Team must win (or lose) by "
                "a certain number of points.  \n"
                "**Total (O/U)** \u2014 Bet on whether the combined "
                "score is over or under a number.  \n"
                "**Arbitrage** \u2014 When different sportsbooks "
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

    return stake_a, stake_b, profit, roi


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
                "commence_time": getattr(ln, "commence_time", None),
            }
        g = games[ln.event]
        g["books"].add(ln.sportsbook)
        g["bet_types"].add(ln.bet_type)
        if ln.timestamp > g["last_updated"]:
            g["last_updated"] = ln.timestamp
        # Prefer non-None commence_time
        if g["commence_time"] is None:
            g["commence_time"] = getattr(ln, "commence_time", None)
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

    _render_legend()

    st.divider()

    # Search filter
    search = st.text_input(
        "Filter by team name",
        key="game_search",
        placeholder="Search teams...",
    )

    # Sort by commence_time ascending (soonest first); missing at bottom
    _far_future = datetime.max.replace(tzinfo=None)
    sorted_games = sorted(
        games.items(),
        key=lambda x: (
            x[1]["commence_time"].astimezone() if x[1]["commence_time"] else _far_future
        ),
    )

    # Apply search filter
    if search:
        q = search.lower()
        sorted_games = [
            (name, info) for name, info in sorted_games
            if q in info["home_team"].lower()
            or q in info["away_team"].lower()
        ]

    if not sorted_games:
        st.info("No games match your search.")
        return

    for event_name, info in sorted_games:
        with st.container(border=True):
            # Time-first layout: time | matchup+badges | books/updated | view
            cols = st.columns([1.2, 5, 2, 1])
            ct = info["commence_time"]
            with cols[0]:
                if ct:
                    st.markdown(f"**{_format_start_time(ct)}**")
                    st.caption(_relative_date_label(ct))
                else:
                    st.markdown("**TBD**")
                    st.caption(f"Updated {_relative_time(info['last_updated'])}")
            with cols[1]:
                badges = []
                if event_name in arb_events:
                    badges.append(":red[**ARB**]")
                elif event_name in near_arb_events:
                    badges.append(":orange[**NEAR-ARB**]")
                if event_name in move_events:
                    badges.append(":blue[**MOVE**]")
                badge_suffix = (
                    "  " + " \u00a0 ".join(badges) if badges else ""
                )
                st.markdown(f"**{event_name}**{badge_suffix}")
                st.caption(sport)
            with cols[2]:
                st.markdown(f"{len(info['books'])} books")
                st.caption(f"Updated {_relative_time(info['last_updated'])}")
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
        st.caption("This is a free call \u2014 does not count against your quota.")
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

        # --- Summary strip: best lines at a glance ---
        _detail_summary_strip(game_lines)

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


# -- Detail: Summary strip -------------------------------------------------

def _detail_summary_strip(game_lines):
    """Compact 4-metric strip showing best values across all markets."""
    ml = [ln for ln in game_lines if ln.bet_type == BetType.MONEYLINE]
    sp = [ln for ln in game_lines if ln.bet_type == BetType.SPREAD]
    tot = [ln for ln in game_lines if ln.bet_type == BetType.TOTAL]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if ml:
            best = max(ml, key=lambda ln: ln.home_value)
            st.metric(
                "Best Home ML",
                _format_odds(best.home_value),
                help=best.sportsbook,
            )
        else:
            st.metric("Best Home ML", "\u2014")

    with c2:
        if ml:
            best = max(ml, key=lambda ln: ln.away_value)
            st.metric(
                "Best Away ML",
                _format_odds(best.away_value),
                help=best.sportsbook,
            )
        else:
            st.metric("Best Away ML", "\u2014")

    with c3:
        if sp:
            best_h = max(sp, key=lambda ln: ln.home_value)
            best_a = max(sp, key=lambda ln: ln.away_value)
            st.metric(
                "Best Spread",
                f"H {best_h.home_value:+.1f} / A {best_a.away_value:+.1f}",
                help=f"{best_h.sportsbook} / {best_a.sportsbook}",
            )
        else:
            st.metric("Best Spread", "\u2014")

    with c4:
        if tot:
            best_o = max(tot, key=lambda ln: ln.home_value)
            best_u = max(tot, key=lambda ln: ln.away_value)
            st.metric(
                "Best Total",
                f"O {best_o.home_value:.1f} / U {best_u.away_value:.1f}",
                help=f"{best_o.sportsbook} / {best_u.sportsbook}",
            )
        else:
            st.metric("Best Total", "\u2014")


# -- Detail tab: Odds Comparison -------------------------------------------

def _detail_odds(event_name: str, game_lines):
    """Shows lines for one selected market, with a market selector."""
    if not game_lines:
        st.info("No odds data for this game. Fetch odds from the dashboard first.")
        return

    # Determine which markets are available
    available_bts: list[BetType] = []
    for bt in [BetType.MONEYLINE, BetType.SPREAD, BetType.TOTAL]:
        if any(ln.bet_type == bt for ln in game_lines):
            available_bts.append(bt)

    if not available_bts:
        st.info("No market data available.")
        return

    available_labels = [BET_TYPE_LABELS[bt.value] for bt in available_bts]

    selected_label = st.radio(
        "Market",
        available_labels,
        horizontal=True,
        key="detail_market_sel",
    )
    selected_bt = available_bts[available_labels.index(selected_label)]

    # Filter to selected market
    bt_lines = [ln for ln in game_lines if ln.bet_type == selected_bt]

    compact = _compact_mode()
    display_df = _lines_to_df(bt_lines, compact=compact)
    raw_df = _lines_to_raw_df(bt_lines)

    # Drop redundant Game and Bet Type columns
    for col in ["Game", "Bet Type"]:
        if col in display_df.columns:
            display_df = display_df.drop(columns=[col])

    if not display_df.empty:
        styled = _highlight_best(display_df, raw_df)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Best-value callouts for selected market
    if selected_bt == BetType.MONEYLINE and len(bt_lines) >= 2:
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
    elif selected_bt == BetType.SPREAD and len(bt_lines) >= 2:
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
    elif selected_bt == BetType.TOTAL and len(bt_lines) >= 2:
        best_over = max(bt_lines, key=lambda ln: ln.home_value)
        best_under = max(bt_lines, key=lambda ln: ln.away_value)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                f"Best Over: {best_over.sportsbook}",
                f"{best_over.home_value:.1f}",
            )
        with c2:
            st.metric(
                f"Best Under: {best_under.sportsbook}",
                f"{best_under.away_value:.1f}",
            )

    # --- Add to Bet Slip controls ---
    _bet_slip_add_controls(event_name, selected_bt, bt_lines)


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
                c3.metric("Profit margin", fmt_pct(arb.margin, sign=True))

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
                                    f"Stake: **{fmt_money(stake_a)}**"
                                )
                            with lc2:
                                st.markdown(
                                    f"**Leg B \u2014 {arb.side_b.sportsbook}**  \n"
                                    f"Bet Away at "
                                    f"{_format_odds(arb.side_b.away_value)}  \n"
                                    f"Stake: **{fmt_money(stake_b)}**"
                                )
                            st.divider()
                            pc1, pc2, pc3 = st.columns(3)
                            pc1.metric("Total wagered", fmt_money(100))
                            pc2.metric("Guaranteed profit", fmt_money(profit))
                            pc3.metric("ROI", fmt_pct(roi))

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
                    "Margin": fmt_pct(arb.margin, sign=True),
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

# Raised default thresholds to reduce noise
_SIG_THRESHOLDS = {
    BetType.MONEYLINE: 5.0,
    BetType.SPREAD: 0.5,
    BetType.TOTAL: 0.5,
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

    # --- Filters ---
    col1, col2 = st.columns(2)
    with col1:
        bt_options = ["All"] + [
            BET_TYPE_LABELS[bt.value]
            for bt in [BetType.MONEYLINE, BetType.SPREAD, BetType.TOTAL]
        ]
        bt_filter = st.selectbox(
            "Bet type", bt_options, key="detail_move_bt",
        )
    with col2:
        time_window = st.selectbox(
            "Time window",
            list(_TIME_WINDOWS.keys()),
            key="detail_move_window",
        )

    col3, col4 = st.columns(2)
    with col3:
        hide_juice = st.toggle(
            "Hide juice-only changes",
            value=True,
            key="detail_hide_juice",
            help=(
                "Hide small changes (ML < 5pts, Spread/Total < 0.5pts). "
                "Turn off to see all movements."
            ),
        )
    with col4:
        latest_only = st.toggle(
            "Latest per book + market",
            value=False,
            key="detail_latest_only",
            help="Show only the most recent move per sportsbook and bet type.",
        )

    # --- Apply filters ---
    filtered = list(game_moves)

    # Bet type filter
    if bt_filter != "All":
        bt_value = _LABEL_TO_BT.get(bt_filter)
        if bt_value:
            filtered = [m for m in filtered if m.bet_type.value == bt_value]

    # Time window filter
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

    # Hide juice-only changes (raised thresholds)
    if hide_juice:
        kept = []
        for m in filtered:
            threshold = _SIG_THRESHOLDS.get(m.bet_type, 0)
            if abs(m.change) >= threshold:
                kept.append(m)
        filtered = kept

    # Dedupe: keep only latest move per sportsbook + bet type
    if latest_only:
        best: dict[tuple, object] = {}
        for m in filtered:
            key = (m.sportsbook, m.bet_type)
            prev = best.get(key)
            if prev is None or m.new_line.timestamp > prev.new_line.timestamp:
                best[key] = m
        filtered = list(best.values())

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
# Bet Slip: add controls (inside detail odds tab)
# ---------------------------------------------------------------------------

def _try_add_leg(leg: dict) -> None:
    """Validate and add a leg to the bet slip, enforcing sportsbook lock."""
    slip = st.session_state.setdefault("bet_slip", [])
    slip_book = _get_slip_book()

    # Enforce single-book constraint
    if slip_book and leg["sportsbook"] != slip_book:
        st.toast(
            f"Slip locked to {slip_book}. Clear slip to change.",
            icon="\u26a0\ufe0f",
        )
        return

    if is_duplicate_leg(slip, leg):
        st.toast("Already in your bet slip!", icon="\u26a0\ufe0f")
        return

    if has_conflicting_leg(slip, leg):
        st.toast(
            "Added \u2014 opposing selection exists for this event!",
            icon="\u26a0\ufe0f",
        )
    else:
        st.toast("Added to bet slip!", icon="\u2705")

    slip.append(leg)
    # Lock to this book on first leg
    if not slip_book:
        _lock_slip_book(leg["sportsbook"])
    st.rerun()


def _bet_slip_add_controls(
    event_name: str, selected_bt: BetType, bt_lines: list,
):
    """Render 'Add to Bet Slip' controls below the odds comparison table."""
    if not bt_lines:
        return

    st.divider()
    st.markdown("**Add to Bet Slip**")

    slip_book = _get_slip_book()
    all_sportsbooks = [ln.sportsbook for ln in bt_lines]

    if selected_bt == BetType.TOTAL:
        sides = ["Over", "Under"]
    else:
        sides = ["Home", "Away"]

    market_key = selected_bt.value  # moneyline / spread / total

    # Determine which books to show in the dropdown
    if slip_book:
        if slip_book in all_sportsbooks:
            available_books = [slip_book]
        else:
            st.info(
                f"Slip locked to **{slip_book}** (not available for this game).  \n"
                "Open Bet Slip to change sportsbook."
            )
            return
    else:
        available_books = all_sportsbooks

    cols = st.columns([3, 2, 1])
    with cols[0]:
        selected_book = st.selectbox(
            "Sportsbook",
            available_books,
            key=f"slip_book_{market_key}",
            label_visibility="collapsed",
        )
    with cols[1]:
        selected_side = st.radio(
            "Side",
            sides,
            horizontal=True,
            key=f"slip_side_{market_key}",
            label_visibility="collapsed",
        )

    if slip_book:
        st.caption(f"Locked to {slip_book}")

    ln = next((x for x in bt_lines if x.sportsbook == selected_book), None)
    if ln is None:
        st.warning(f"No line found for {selected_book}")
        return

    # Determine odds and line based on market + side
    odds: float | None = None
    line: float | None = None
    if selected_bt == BetType.MONEYLINE:
        odds = ln.home_value if selected_side == "Home" else ln.away_value
        line = None
        market_label = "ML"
    elif selected_bt == BetType.SPREAD:
        if selected_side == "Home":
            odds = ln.home_price
            line = ln.home_value
        else:
            odds = ln.away_price
            line = ln.away_value
        market_label = "Spread"
    else:  # TOTAL
        if selected_side == "Over":
            odds = ln.home_price
            line = ln.home_value
        else:
            odds = ln.away_price
            line = ln.away_value
        market_label = "Total"

    if odds is None:
        st.caption("Juice/price data not available for this selection.")
        return

    # Preview what will be added
    if line is not None and market_label == "Spread":
        line_str = f" {line:+.1f}"
    elif line is not None:
        line_str = f" {line:.1f}"
    else:
        line_str = ""
    st.caption(
        f"{selected_book} \u00b7 {market_label} \u00b7 "
        f"{selected_side}{line_str} \u00b7 {_format_odds(odds)}"
    )

    with cols[2]:
        if st.button(
            "\u2795 Slip", key=f"slip_add_{market_key}", type="primary",
        ):
            leg = {
                "sport": _sport_name(),
                "event_name": event_name,
                "sportsbook": selected_book,
                "market": market_label,
                "selection": selected_side,
                "line": line,
                "odds": odds,
                "fetched_at": ln.timestamp.isoformat(),
            }
            _try_add_leg(leg)


# ---------------------------------------------------------------------------
# Bet Slip: dialog (modal popup)
# ---------------------------------------------------------------------------

@st.dialog("Bet Slip", width="large")
def _slip_dialog():
    """Sportsbook-style bet slip popup."""
    slip = st.session_state.get("bet_slip", [])
    slip_book = _get_slip_book()

    if not slip:
        st.info("No legs yet. Add selections from the Odds Comparison tab.")
        if st.button("Close", key="dlg_close_empty"):
            st.rerun()
        return

    # Locked sportsbook header
    hcols = st.columns([4, 2])
    with hcols[0]:
        if slip_book:
            st.markdown(f"**Locked to {slip_book}**")
    with hcols[1]:
        if st.button("Change sportsbook", key="dlg_change_book"):
            _clear_slip()
            st.rerun()

    st.divider()

    # --- Legs list ---
    to_remove: int | None = None
    for i, leg in enumerate(slip):
        with st.container(border=True):
            lcols = st.columns([5, 1])
            with lcols[0]:
                if leg["line"] is not None and leg["market"] == "Spread":
                    line_str = f" {leg['line']:+.1f}"
                elif leg["line"] is not None:
                    line_str = f" {leg['line']:.1f}"
                else:
                    line_str = ""

                st.markdown(
                    f"**{leg['event_name']}**  \n"
                    f"{leg['market']} \u00b7 {leg['selection']}{line_str}  \n"
                    f"{leg['sportsbook']} \u00b7 "
                    f"**{format_american(leg['odds'])}**"
                )
            with lcols[1]:
                if st.button(
                    "\u2715", key=f"dlg_rm_{i}", help="Remove leg",
                ):
                    to_remove = i

    if to_remove is not None:
        slip.pop(to_remove)
        if not slip:
            st.session_state["slip_book"] = None
        st.session_state["_slip_reopen"] = True
        st.rerun()

    # Clear all
    if len(slip) > 1:
        if st.button("Clear All", key="dlg_clear"):
            _clear_slip()
            st.session_state["_slip_reopen"] = True
            st.rerun()

    st.divider()

    # --- Payout section ---
    legs_odds = [leg["odds"] for leg in slip]
    # Sanitize: delete invalid widget key so Streamlit can't use it.
    _ds = st.session_state.get("dlg_stake")
    if not isinstance(_ds, (int, float)) or _ds < 1.0:
        st.session_state.pop("dlg_stake", None)
    _slip = st.session_state.get("slip_stake", 100.0)
    _default = max(1.0, float(_slip)) if isinstance(_slip, (int, float)) else 100.0
    stake = st.number_input(
        "Stake ($)",
        min_value=1.0,
        value=_default,
        step=10.0,
        key="dlg_stake",
    )
    # Sync to main slip_stake
    st.session_state["slip_stake"] = stake

    if len(slip) == 1:
        odds = legs_odds[0]
        profit = american_profit(stake, odds)
        total_ret = american_total_return(stake, odds)
        dec = _slip_a2d(odds)

        st.markdown("**Straight Bet**")
        c1, c2 = st.columns(2)
        c1.metric("Odds", f"{format_american(odds)} ({dec:.2f})")
        c2.metric("Profit", fmt_money(profit))
        st.metric("Total Payout", fmt_money(total_ret))
    else:
        result = parlay_payout(stake, legs_odds)
        st.markdown(f"**{len(slip)}-Leg Parlay**")
        c1, c2 = st.columns(2)
        c1.metric(
            "Combined Odds",
            f"{format_american(result['combined_american'])} "
            f"({result['combined_decimal']:.2f})",
        )
        c2.metric("Profit", fmt_money(result['profit']))
        st.metric("Total Payout", fmt_money(result['total_return']))

    st.divider()

    # Action buttons
    bcols = st.columns(2)
    with bcols[0]:
        if st.button(
            "Mock Submit", key="dlg_submit", type="primary",
            use_container_width=True,
        ):
            try:
                submit_bet(st.session_state, stake)
                st.session_state["_slip_submitted"] = True
            except ValueError as exc:
                st.error(str(exc))
            st.rerun()
    with bcols[1]:
        if st.button("Close", key="dlg_close", use_container_width=True):
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE 3: Best Lines to Shop
# ---------------------------------------------------------------------------

def _page_best_lines():
    """Rank lines by shopping value (NOT predicting winners)."""
    st.title("Best Lines to Shop")
    st.caption(
        "Find where one sportsbook offers significantly better odds "
        "than the consensus. Positive edge = better-than-median value."
    )

    _render_legend()

    lines = st.session_state.get("last_fetch")
    if not lines:
        st.info(
            "No data loaded yet. Fetch odds from the Dashboard first."
        )
        return

    # Filters
    fcols = st.columns(4)
    with fcols[0]:
        market_filter = st.selectbox(
            "Market",
            ["All", "ML", "Spread", "Total"],
            key="bl_market",
        )
    with fcols[1]:
        min_edge = st.number_input(
            "Min edge (%)",
            min_value=0.0,
            value=0.5,
            step=0.25,
            key="bl_min_edge",
            help="Minimum edge in percentage points vs median.",
        )
    with fcols[2]:
        max_rows = st.select_slider(
            "Max rows",
            options=[10, 15, 20, 25, 50],
            value=25,
            key="bl_max_rows",
        )
    with fcols[3]:
        # Sanitize: delete invalid widget key so Streamlit can't use it.
        _bl = st.session_state.get("bl_stake")
        if not isinstance(_bl, (int, float)) or _bl < 1.0:
            st.session_state.pop("bl_stake", None)
        _slip = st.session_state.get("slip_stake", 100.0)
        _default = max(1.0, float(_slip)) if isinstance(_slip, (int, float)) else 100.0
        stake = st.number_input(
            "Stake ($)",
            min_value=1.0,
            value=_default,
            step=10.0,
            key="bl_stake",
        )
        st.session_state["slip_stake"] = stake

    # Compute standouts
    entries = _lines_to_shopping_entries(lines)
    standouts = compute_standouts(entries, stake=stake)

    # Apply filters
    if market_filter != "All":
        standouts = [s for s in standouts if s["market"] == market_filter]

    edge_threshold = min_edge / 100.0
    standouts = [s for s in standouts if s["edge"] >= edge_threshold]
    standouts = standouts[:max_rows]

    if not standouts:
        st.info(
            "No standouts match the current filters. "
            "Try lowering the edge threshold."
        )
        return

    st.markdown(f"**{len(standouts)} standout{'s' if len(standouts) != 1 else ''}**")

    slip_book = _get_slip_book()

    # Build event → commence_time map from fetched lines
    commence_map: dict[str, datetime] = {}
    for ln in lines:
        ct = getattr(ln, "commence_time", None)
        if ct is not None and ln.event not in commence_map:
            commence_map[ln.event] = ct

    # Group standouts by local date
    today = date.today()
    tomorrow = today + timedelta(days=1)

    groups: dict[date | None, list[tuple[int, dict]]] = defaultdict(list)
    for rank, s in enumerate(standouts, 1):
        ct = commence_map.get(s["event"])
        if ct is not None:
            local_date = ct.astimezone().date()
        else:
            local_date = None
        groups[local_date].append((rank, s))

    # Sort within each group by commence_time ascending, then edge descending
    _far_future = datetime.max.replace(tzinfo=None)
    for items in groups.values():
        items.sort(key=lambda x: (
            commence_map.get(x[1]["event"], _far_future),
            -x[1]["edge"],
        ))

    # Order: dated groups sorted ascending, None ("Unknown time") last
    dated_keys: list[date] = sorted(k for k in groups if k is not None)
    ordered_keys: list[date | None] = list(dated_keys)
    if None in groups:
        ordered_keys.append(None)

    for group_date in ordered_keys:
        count = len(groups[group_date])
        date_str = (
            group_date.strftime("%a %b %-d") if group_date is not None else ""
        )
        if group_date is None:
            header = f"Unknown time ({count})"
        elif group_date == today:
            header = f"Today \u2014 {date_str} ({count})"
        elif group_date == tomorrow:
            header = f"Tomorrow \u2014 {date_str} ({count})"
        else:
            header = f"{date_str} ({count})"

        st.subheader(header)

        for rank, s in groups[group_date]:
            with st.container(border=True):
                rcols = st.columns([0.5, 3, 1.5, 1.5, 1.5, 1.5, 2])
                with rcols[0]:
                    st.markdown(f"**{rank}**")
                with rcols[1]:
                    line_str = ""
                    if s["line"] is not None and s["market"] == "Spread":
                        line_str = f" ({s['line']:+.1f})"
                    elif s["line"] is not None:
                        line_str = f" ({s['line']:.1f})"
                    ev_ct = commence_map.get(s["event"])
                    time_str = (
                        f" \u00b7 {_format_start_time(ev_ct)}"
                        if ev_ct else ""
                    )
                    st.markdown(
                        f"**{s['event']}**  \n"
                        f"{s['market']} \u00b7 {s['selection']}{line_str}{time_str}"
                    )
                with rcols[2]:
                    st.markdown(
                        f"**{s['sportsbook']}**  \n"
                        f"{format_american(s['odds'])}"
                    )
                with rcols[3]:
                    st.markdown(
                        f"Median  \n"
                        f"{format_american(s['median_odds'])}"
                    )
                with rcols[4]:
                    edge_pp = s["edge"] * 100
                    st.metric("Edge", fmt_pct(edge_pp, sign=True))
                with rcols[5]:
                    st.metric(
                        "$ Impact",
                        fmt_money(s['dollar_impact'], sign=True),
                    )
                with rcols[6]:
                    # View game button
                    if st.button("View", key=f"bl_view_{rank}"):
                        st.session_state["page"] = "detail"
                        st.session_state["selected_game"] = s["event"]
                        st.rerun()

                    # Add to slip (respects lock)
                    can_add = not slip_book or s["sportsbook"] == slip_book
                    if can_add:
                        if st.button(
                            "\u2795 Slip", key=f"bl_add_{rank}", type="secondary",
                        ):
                            leg = {
                                "sport": _sport_name(),
                                "event_name": s["event"],
                                "sportsbook": s["sportsbook"],
                                "market": s["market"],
                                "selection": s["selection"],
                                "line": s["line"],
                                "odds": s["odds"],
                                "fetched_at": "",
                            }
                            _try_add_leg(leg)
                    elif slip_book:
                        st.caption(f"Locked to {slip_book}")


# ---------------------------------------------------------------------------
# Bet History dialog
# ---------------------------------------------------------------------------

@st.dialog("Bet History", width="large")
def _bet_history_dialog():
    """Modal showing Active and Settled bets."""
    init_bet_state(st.session_state)
    active: list = st.session_state["active_bets"]
    settled: list = st.session_state["settled_bets"]

    tab_active, tab_settled = st.tabs([
        f"Active ({len(active)})",
        f"Settled ({len(settled)})",
    ])

    with tab_active:
        if not active:
            st.info("No active bets. Submit a bet from the Bet Slip.")
        for bet in active:
            with st.container(border=True):
                hcols = st.columns([4, 2, 2])
                with hcols[0]:
                    legs_desc = ", ".join(
                        f"{lg['market']} {lg['selection']}"
                        for lg in bet.legs
                    )
                    st.markdown(
                        f"**{bet.sportsbook}** — "
                        f"{len(bet.legs)} leg{'s' if len(bet.legs) != 1 else ''}  \n"
                        f"{legs_desc}"
                    )
                    for lg in bet.legs:
                        line_str = ""
                        if lg.get("line") is not None and lg["market"] == "Spread":
                            line_str = f" {lg['line']:+.1f}"
                        elif lg.get("line") is not None:
                            line_str = f" {lg['line']:.1f}"
                        st.caption(
                            f"{lg['event_name']} · {lg['market']} · "
                            f"{lg['selection']}{line_str} · "
                            f"{format_american(lg['odds'])}"
                        )
                with hcols[1]:
                    st.metric("Stake", fmt_money(bet.stake))
                    st.metric(
                        "Odds",
                        fmt_odds(bet.combined_american),
                    )
                with hcols[2]:
                    st.metric("Potential Payout", fmt_money(bet.total_payout))
                    st.caption(f"Placed {bet.created_at[:16]}")

                # Settlement controls
                st.markdown("**Settle this bet:**")
                scols = st.columns(3)
                with scols[0]:
                    if st.button(
                        "Won", key=f"hist_won_{bet.id}",
                        type="primary", use_container_width=True,
                    ):
                        settle_bet(st.session_state, bet.id, "won")
                        st.session_state["_reopen_history"] = True
                        st.rerun()
                with scols[1]:
                    if st.button(
                        "Lost", key=f"hist_lost_{bet.id}",
                        use_container_width=True,
                    ):
                        settle_bet(st.session_state, bet.id, "lost")
                        st.session_state["_reopen_history"] = True
                        st.rerun()
                with scols[2]:
                    if st.button(
                        "Push", key=f"hist_push_{bet.id}",
                        use_container_width=True,
                    ):
                        settle_bet(st.session_state, bet.id, "push")
                        st.session_state["_reopen_history"] = True
                        st.rerun()

    with tab_settled:
        if not settled:
            st.info("No settled bets yet.")
        else:
            # Summary metrics
            total_staked = sum(b.stake for b in settled)
            total_profit = sum(
                b.total_payout - b.stake if b.status == "won"
                else (0.0 if b.status == "push" else -b.stake)
                for b in settled
            )
            wins = sum(1 for b in settled if b.status == "won")
            losses = sum(1 for b in settled if b.status == "lost")
            pushes = sum(1 for b in settled if b.status == "push")

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Staked", fmt_money(total_staked))
            mc2.metric("Net Profit", fmt_money(total_profit, sign=True))
            mc3.metric("Record", f"{wins}W-{losses}L-{pushes}P")
            roi = (total_profit / total_staked * 100) if total_staked else 0
            mc4.metric("ROI", fmt_pct(roi, sign=True))

            st.divider()

            for bet in reversed(settled):
                with st.container(border=True):
                    status_icon = {
                        "won": ":green[**WON**]",
                        "lost": ":red[**LOST**]",
                        "push": ":orange[**PUSH**]",
                    }.get(bet.status, bet.status)

                    hcols = st.columns([4, 2, 2])
                    with hcols[0]:
                        legs_desc = ", ".join(
                            f"{lg['market']} {lg['selection']}"
                            for lg in bet.legs
                        )
                        n = len(bet.legs)
                        sfx = "s" if n != 1 else ""
                        st.markdown(
                            f"{status_icon} — "
                            f"**{bet.sportsbook}** — "
                            f"{n} leg{sfx}  \n"
                            f"{legs_desc}"
                        )
                    with hcols[1]:
                        st.metric("Stake", fmt_money(bet.stake))
                        st.metric(
                            "Odds",
                            fmt_odds(bet.combined_american),
                        )
                    with hcols[2]:
                        if bet.status == "won":
                            pnl = bet.total_payout - bet.stake
                            st.metric("Profit", f":green[+{fmt_money(pnl)}]")
                        elif bet.status == "push":
                            st.metric("Profit", fmt_money(0))
                        else:
                            st.metric("Profit", f":red[-{fmt_money(bet.stake)}]")
                        if bet.settled_at:
                            st.caption(f"Settled {bet.settled_at[:16]}")

    st.divider()
    if st.button("Close", key="hist_close", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------------------------
# Legend component
# ---------------------------------------------------------------------------

def _render_legend():
    """Render an on-screen legend explaining all highlight colors and badges."""
    with st.expander("Legend — Colors & Badges", expanded=False):
        st.markdown(
            "| Symbol | Meaning |\n"
            "|---|---|\n"
            "| :green-background[**Best Price**] | "
            "Green cell = best odds for that side among all sportsbooks |\n"
            "| :red[**ARB**] | "
            "Profitable arbitrage — guaranteed profit by betting both sides |\n"
            "| :orange[**NEAR-ARB**] | "
            "Close to arbitrage (margin > −2%) — worth monitoring |\n"
            "| :blue[**MOVE**] | "
            "Significant line movement detected since last fetch |\n"
            "| **Edge %** | "
            "How much better a line is vs the median across books |\n"
            "| **$ Impact** | "
            "Extra payout vs median book for your stake (Best Lines page) |"
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

    # Ensure bet slip exists in session state
    if "bet_slip" not in st.session_state:
        st.session_state["bet_slip"] = []
    if "slip_book" not in st.session_state:
        st.session_state["slip_book"] = None
    if st.session_state.get("slip_stake", 0.0) < 1.0:
        st.session_state["slip_stake"] = 100.0

    # Ensure bet history state exists
    init_bet_state(st.session_state)

    # Toast for mock submit
    if st.session_state.pop("_slip_submitted", False):
        st.toast("Bet submitted!", icon="\u2705")

    _sidebar()

    # Open bet slip dialog (from button or reopen after state change)
    if st.session_state.pop("_open_slip", False):
        _slip_dialog()
    elif st.session_state.pop("_slip_reopen", False):
        _slip_dialog()

    # Open bet history dialog
    if st.session_state.pop("_open_history", False):
        _bet_history_dialog()
    elif st.session_state.pop("_reopen_history", False):
        _bet_history_dialog()

    page = st.session_state.get("page", "dashboard")
    if page == "detail" and st.session_state.get("selected_game"):
        _page_detail()
    else:
        nav = st.session_state.get("nav_page", "Dashboard")
        if nav == "Best Lines to Shop":
            _page_best_lines()
        else:
            _page_dashboard()


if __name__ == "__main__":
    main()

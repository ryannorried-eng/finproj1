"""Streamlit web dashboard — run with: streamlit run src/line_tracker/dashboard.py"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from line_tracker.arbitrage import find_moneyline_arbs, find_spread_arbs
from line_tracker.best_bets import recommend_best_bets
from line_tracker.bet_history import (
    close_bet_clv,
    compute_clv,
    init_bet_state,
    load_bets_from_db,
    persist_bet,
    settle_bet,
    settle_bet_persistent,
    snapshot_pick,
    submit_bet,
)
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
from line_tracker.calibration import (
    calibrate_thresholds,
    calibration_from_json,
    calibration_to_json,
    format_calibration_report,
    load_clv_training_df,
)
from line_tracker.market_structure import analyze_market
from line_tracker.models import BetType
from line_tracker.movements import detect_moves
from line_tracker.performance import (
    all_breakdowns,
    apply_filters,
    build_clv_dataframe,
    calibration_stats,
    clv_color,
    clv_distribution,
    rolling_clv_series,
    summary_kpis,
)
from line_tracker.scraper import OddsClient
from line_tracker.slate import (
    build_daily_slate,
    passes_relaxed_tier2,
    thresholds_from_calibration,
)
from line_tracker.storage import DEFAULT_DB_PATH, LineStore

# Toggle to show EV Math Debug expander on slate / shopping pages.
SHOW_EV_DEBUG = False

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

# Sentinel used to sort games with missing commence_time to the bottom.
_FAR_FUTURE = datetime.max.replace(tzinfo=None)

DB_PATH = str(DEFAULT_DB_PATH)


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


def _bankroll() -> float:
    """Return the user's bankroll setting, or 0 if unset."""
    return st.session_state.get("bankroll", 0.0)


def _kelly_line(rec, prefix: str = "Sizing") -> str:
    """Build a markdown string showing Kelly sizing info for *rec*.

    If a bankroll is set, also shows the dollar stake.
    """
    if rec.kelly_suggested <= 0:
        return ""
    pct = rec.kelly_suggested * 100
    br = _bankroll()
    if br > 0:
        stake = br * rec.kelly_suggested
        return (
            f"{prefix}: **{pct:.1f}%** "
            f"({fmt_money(stake)}) — {rec.sizing_note}"
        )
    return f"{prefix}: **{pct:.1f}%** — {rec.sizing_note}"


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
        try:
            with LineStore() as _db:
                active_count = len(_db.get_bets(status="active"))
                settled_count = len(
                    [b for b in _db.get_bets() if b["status"] != "active"],
                )
        except Exception:
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
            ["Dashboard", "Best Lines to Shop", "Daily Slate", "Performance"],
            key="nav_page",
            horizontal=True,
        )

        st.divider()

        st.header("Settings")

        st.text_input(
            "API Key",
            value=os.environ.get("ODDS_API_KEY", "09d11879822c9c7bd81c7eb210c82d92"),
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

        st.number_input(
            "Bankroll ($)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            key="bankroll",
            help=(
                "Enter your total bankroll to see Kelly-based "
                "suggested stake amounts alongside recommendations."
            ),
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
            st.toggle(
                "Slate debug counters",
                value=False,
                key="slate_debug",
                help="Show classification breakdown on the Daily Slate page.",
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
    sorted_games = sorted(
        games.items(),
        key=lambda x: (
            x[1]["commence_time"].astimezone() if x[1]["commence_time"] else _FAR_FUTURE
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

        # --- Best Bet (Consensus EV) ---
        _detail_best_bet_section(game_lines)

        # --- Market Structure ---
        _detail_market_structure(game_lines)

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


# -- Detail: Best Bet (Consensus EV) ---------------------------------------

def _detail_best_bet_section(game_lines):
    """Show the 'Best Bet (Consensus EV)' section above tabs."""
    recs = recommend_best_bets(game_lines, top_n=6)
    if not recs:
        return

    st.divider()
    st.subheader("Best Bet (Market Consensus EV)")
    st.caption(
        "Based on vig-free consensus probabilities from the books \u00b7 "
        "Informational only, not financial advice"
    )

    slider_cols = st.columns(2)
    with slider_cols[0]:
        min_edge = st.slider(
            "Min edge (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            key=f"min_edge_{id(game_lines)}",
        )
    with slider_cols[1]:
        min_quality = st.slider(
            "Min quality",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            key=f"min_quality_{id(game_lines)}",
        )

    qualified = [
        r for r in recs
        if r.edge_pct >= min_edge and r.quality_score >= min_quality
    ]

    if qualified:
        top = qualified[0]
        market_label = _best_bet_market_label(top)

        with st.container(border=True):
            st.markdown(
                f"**Best Bet: {top.selection} ({market_label}) "
                f"at {format_american(top.best_odds)} "
                f"on {top.best_sportsbook}**"
            )
            _top_ev = fmt_money(top.ev_per_100, sign=True).replace("$", r"\$")
            st.markdown(
                f"EV: **{_top_ev} per \\$100** | "
                f"Edge Z: **{top.edge_z:+.2f}** | "
                f"Consensus (weighted): "
                f"**{top.consensus_prob_weighted * 100:.1f}%** | "
                f"Breakeven: **{top.breakeven_prob * 100:.1f}%**"
            )
            st.markdown(
                f"Market confidence: **{top.confidence}** | "
                f"Quality: **{top.quality_score}/100 ({top.quality_tier})**",
                help=(
                    "Confidence measures sportsbook disagreement. "
                    "Quality is a composite score of edge, agreement, "
                    "coverage, and data freshness."
                ),
            )

            kelly_str = _kelly_line(top)
            if kelly_str:
                st.markdown(kelly_str)

            _render_best_bet_why(top)

        others = qualified[1:3]
        if others:
            st.markdown("**Other +EV bets:**")
            for r in others:
                ml = _best_bet_market_label(r)
                ev_str = fmt_money(r.ev_per_100, sign=True).replace("$", "\\$")
                st.markdown(
                    f"- {r.selection} ({ml}) at "
                    f"{format_american(r.best_odds)} "
                    f"on {r.best_sportsbook} — "
                    f"EV: {ev_str} per \\$100, "
                    f"Z: {r.edge_z:+.2f} — "
                    f"Quality: {r.quality_score} ({r.quality_tier})"
                )
    else:
        st.info(
            "No bets meet your edge + quality thresholds."
            " Showing closest candidates:"
        )
        for r in recs[:3]:
            ml = _best_bet_market_label(r)
            ev_str = fmt_money(r.ev_per_100, sign=True).replace("$", "\\$")
            st.markdown(
                f"- {r.selection} ({ml}) at "
                f"{format_american(r.best_odds)} "
                f"on {r.best_sportsbook} — "
                f"EV: {ev_str} per \\$100, "
                f"Z: {r.edge_z:+.2f} — "
                f"Quality: {r.quality_score} ({r.quality_tier})"
            )


def _render_best_bet_why(rec) -> None:
    """Render a compact 'Why this bet?' breakdown inside the Best Bet card."""
    with st.expander("Why this bet?", expanded=False):
        books_line = ""
        if rec.books_used_count and rec.total_books_count:
            line_label = ""
            if rec.line is not None:
                line_label = f" at line {rec.line:g}"
            books_line = (
                f"- **Consensus built from** "
                f"{rec.books_used_count} / {rec.total_books_count} "
                f"books{line_label}\n"
            )
        ev_str = fmt_money(rec.ev_per_100, sign=True).replace("$", "\\$")
        st.markdown(
            f"{books_line}"
            f"- **Consensus (weighted, vig-free):** "
            f"{rec.consensus_prob_weighted * 100:.1f}%\n"
            f"- **Unweighted consensus (vig-free):** "
            f"{rec.unweighted_consensus_prob * 100:.1f}%\n"
            f"- **Newest book update:** {rec.newest_update_age_min:.0f}m ago\n"
            f"- **Oldest book update:** {rec.oldest_update_age_min:.0f}m ago\n"
            f"- **Best price:** {format_american(rec.best_odds)} "
            f"at {rec.best_sportsbook}\n"
            f"- **Breakeven prob (at that price):** "
            f"{rec.breakeven_prob * 100:.1f}%\n"
            f"- **EV:** {ev_str} per \\$100\n"
            f"- **Edge Z:** {rec.edge_z:+.2f} "
            f"(confidence: {rec.confidence})\n"
            f"- **n_eff:** {rec.n_eff:.1f} "
            f"(books: {rec.books_used_count})\n"
            f"- **Quality:** {rec.quality_score}/100 ({rec.quality_tier})"
        )
        st.caption(
            f"Edge Score: {rec.edge_score:.0f} | "
            f"Agreement: {rec.agreement_score:.0f} | "
            f"Coverage: {rec.coverage_score:.0f} | "
            f"Freshness: {rec.freshness_score:.0f}"
        )

        # Market volatility & hold metrics
        st.markdown(
            f"- **Market hold median:** {rec.market_hold_median:.2f}%\n"
            f"- **Sigma (prob):** {rec.robust_sigma:.4f} | "
            f"**Sigma (EV):** {rec.ev_sigma:.4f}\n"
            f"- **Edge EV:** {rec.edge_ev:+.6f} | "
            f"**Shrunk:** {rec.edge_ev_shrunk:+.6f}"
        )

        if rec.book_holds:
            holds_str = " | ".join(
                f"{book}: {hold:.2f}%"
                for book, hold in sorted(
                    rec.book_holds.items(), key=lambda kv: kv[1]
                )
            )
            st.caption(f"Book holds: {holds_str}")


def _best_bet_market_label(rec) -> str:
    """Format the market name with line info for display."""
    if rec.market == "spread" and rec.line is not None:
        return f"Spread ({rec.line:+.1f})"
    if rec.market == "total" and rec.line is not None:
        return f"Total ({rec.line:.1f})"
    return rec.market.title()


# -- Detail: Market Structure -----------------------------------------------

_TAG_COLORS = {
    "Efficient": "green",
    "Normal": "blue",
    "Noisy": "orange",
}

_MKT_LABELS = {
    "moneyline": "Moneyline",
    "spread": "Spread",
    "total": "Total",
}


def _detail_market_structure(game_lines):
    """Show a Market Structure card with efficiency metrics."""
    ml = [ln for ln in game_lines if ln.bet_type == BetType.MONEYLINE]
    sp = [ln for ln in game_lines if ln.bet_type == BetType.SPREAD]
    tot = [ln for ln in game_lines if ln.bet_type == BetType.TOTAL]

    analyses = {}
    for label, subset in [("moneyline", ml), ("spread", sp), ("total", tot)]:
        a = analyze_market(subset)
        if a:
            analyses[label] = a

    if not analyses:
        return

    st.divider()
    st.subheader("Market Structure")
    st.caption("Efficiency and stability metrics across sportsbooks")

    for mkt_key, info in analyses.items():
        mkt_name = _MKT_LABELS.get(mkt_key, mkt_key)
        tag = info["tag"]
        tag_color = _TAG_COLORS.get(tag, "gray")

        with st.container(border=True):
            hdr = (
                f"**{mkt_name}** — "
                f":{tag_color}-background[**{tag}**]"
            )
            st.markdown(hdr)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Books",
                f"{info['books_with_holds']}/{info['books_total']}",
            )
            c2.metric("Hold (median)", f"{info['market_hold_median']:.1f}%")
            h_spr = info["hold_spread"]
            c3.metric(
                "Hold Spread",
                f"{h_spr:.1f}%" if h_spr is not None else "N/A",
                help="IQR (p75-p25) of book hold percentages",
            )
            c4.metric(
                "Volatility",
                f"{info['volatility_sigma']:.4f}",
                help="Std dev of de-vigged probabilities",
            )

            # Sharp vs retail divergence
            div = info["divergence"]
            if div is not None:
                d1, d2, d3 = st.columns(3)
                d1.metric(
                    "Sharp Consensus",
                    f"{info['sharp_consensus'] * 100:.1f}%",
                )
                d2.metric(
                    "Retail Consensus",
                    f"{info['retail_consensus'] * 100:.1f}%",
                )
                d3.metric(
                    "Divergence",
                    f"{div * 100:.1f} pp",
                    help=(
                        "Absolute difference between sharp and "
                        "retail consensus probabilities"
                    ),
                )
                sharp_str = ", ".join(info["sharp_books"]) or "—"
                retail_str = ", ".join(info["retail_books"]) or "—"
                st.caption(
                    f"Sharp: {sharp_str} | Retail: {retail_str}"
                )
            else:
                st.caption(
                    "Sharp/retail divergence: insufficient books "
                    "in one or both groups"
                )

            # Line dispersion (spread/total)
            disp = info.get("line_dispersion")
            if disp:
                parts = [
                    f"{val:g} ({cnt})"
                    for val, cnt in disp.items()
                ]
                st.caption(f"Line dispersion: {', '.join(parts)}")


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
                bet = submit_bet(st.session_state, stake)
                st.session_state["_slip_submitted"] = True
                try:
                    with LineStore() as _s:
                        persist_bet(bet, _s)
                        snapshot_pick(bet, _s)
                except Exception:
                    pass  # DB + CLV snapshot is best-effort
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

    _render_ev_debug_shopping(standouts)

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
    for items in groups.values():
        items.sort(key=lambda x: (
            commence_map.get(x[1]["event"], _FAR_FUTURE),
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
# PAGE 4: Daily Slate
# ---------------------------------------------------------------------------

def _page_daily_slate():
    """Aggregated daily slate with tier-based recommendations."""
    st.title("Daily Slate")
    st.caption(
        "Today's best bets across all events, ranked and tiered by "
        "consensus edge and quality score."
    )

    lines = st.session_state.get("last_fetch")
    if not lines:
        st.info("No data loaded yet. Fetch odds from the Dashboard first.")
        return

    # --- Display Filters ---
    fcols = st.columns(3)
    with fcols[0]:
        min_edge = st.slider(
            "Min edge (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            key="slate_min_edge",
        )
    with fcols[1]:
        min_quality = st.slider(
            "Min quality",
            min_value=0,
            max_value=100,
            value=65,
            step=5,
            key="slate_min_quality",
        )
    with fcols[2]:
        hide_low = st.toggle(
            "Hide low confidence",
            value=True,
            key="slate_hide_low",
        )

    # --- Slate Filters card ---
    _debug_env = os.environ.get(
        "LINE_TRACKER_DEBUG", ""
    ).lower() in ("1", "true", "yes")

    with st.expander("Slate Filters", expanded=False):
        sf1, sf2 = st.columns(2)
        with sf1:
            mode = st.radio(
                "Mode",
                options=["Standard", "Pro", "Auto"],
                index=0,
                key="slate_mode",
                help=(
                    "**Standard**: Top Plays = Tier 1A + 1B. "
                    "**Pro**: Top Plays = Tier 1A only. "
                    "**Auto**: Use CLV-calibrated thresholds."
                ),
                horizontal=True,
            )
            pro_mode = mode == "Pro"
            strict_mode = st.toggle(
                "Strict mode",
                value=True,
                key="slate_strict_mode",
                help=(
                    "ON: Tier 2 uses strict thresholds (positive EV). "
                    "OFF: relaxed (EV >= $0.30, edge_z >= 0.75, "
                    "Low confidence allowed if quality tier >= Strong). "
                    "Tier 1 is always strict."
                ),
            )
            show_stay_away = st.checkbox(
                "Show Stay Away",
                value=True,
                key="slate_show_stay_away",
            )
            show_closest = st.checkbox(
                "Show Closest to Top Plays when none qualify",
                value=True,
                key="slate_show_closest",
            )
        with sf2:
            max_per_section = st.slider(
                "Max games per section",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                key="slate_max_per_section",
            )
            _sort_options = [
                "Best EV",
                "Best edge_z",
                "Best quality",
                "Lowest hold",
            ]
            tier2_sort = st.selectbox(
                "Tier 2 sort",
                options=_sort_options,
                index=0,
                key="slate_tier2_sort",
            )
        if _debug_env:
            show_debug_counts = st.checkbox(
                "Show debug counts",
                value=False,
                key="slate_show_debug_counts",
            )
        else:
            show_debug_counts = False

    # --- Build lines_by_event ---
    lines_by_event: dict[str, list] = defaultdict(list)
    for ln in lines:
        lines_by_event[ln.event].append(ln)

    # --- Resolve thresholds for Auto mode ---
    auto_thresholds = None
    if mode == "Auto":
        store = LineStore(DB_PATH)
        cal_json = store.load_calibration("global")
        if cal_json:
            cal_dict = calibration_from_json(cal_json)
            auto_thresholds = thresholds_from_calibration(cal_dict)
            st.info(
                "Using calibrated thresholds (global). "
                "Run Calibrate on the Performance page to update."
            )
        else:
            st.warning(
                "No calibration found — using Standard defaults. "
                "Run Calibrate on the Performance page first."
            )

    show_debug = show_debug_counts or st.session_state.get("slate_debug", False)
    filters = {
        "min_edge": min_edge,
        "min_quality": min_quality,
        "hide_low_confidence": hide_low,
        "max_per_event": 2,
        "debug": show_debug,
    }
    slate = build_daily_slate(
        dict(lines_by_event),
        filters=filters,
        thresholds=auto_thresholds,
    )

    # ── Assemble display tiers based on mode ──────────────────────────
    if pro_mode:
        top_plays = list(slate["tier1a"])
        more_plays = list(slate["tier1b"]) + list(slate["tier2"])
    else:
        top_plays = list(slate["tier1"])
        more_plays = list(slate["tier2"])

    stay_away = list(slate["stay_away"])
    closest = list(slate["closest_candidates"])
    counts = slate["counts"]

    # ── Relaxed mode: promote qualifying stay_away → display more ─────
    if not strict_mode:
        promoted = [e for e in stay_away if passes_relaxed_tier2(e)]
        stay_away = [e for e in stay_away if not passes_relaxed_tier2(e)]
        more_plays = more_plays + promoted

    # ── Tier 2 sorting (stable, deterministic) ────────────────────────
    _tier2_sort_keys = {
        "Best EV": lambda e: (-e["edge_pct"], -e["slate_score"], e["event"]),
        "Best edge_z": lambda e: (-e.get("edge_z", 0.0), -e["slate_score"], e["event"]),
        "Best quality": lambda e: (-e["quality_score"], -e["slate_score"], e["event"]),
        "Lowest hold": lambda e: (
            e.get("market_hold_median", 999.0),
            -e["slate_score"],
            e["event"],
        ),
    }
    more_plays.sort(key=_tier2_sort_keys.get(tier2_sort, _tier2_sort_keys["Best EV"]))

    # ── Apply max-per-section cap ─────────────────────────────────────
    top_display = top_plays[:max_per_section]
    more_display = more_plays[:max_per_section]
    stay_away_display = stay_away[:max_per_section] if show_stay_away else []

    # ── No-data guard ─────────────────────────────────────────────────
    if counts["total_recs"] == 0:
        st.warning(
            "No recommendations returned "
            "(check API key, sport selection, or fetch)."
        )

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Top Plays", len(top_display))
    mc2.metric("More Plays", len(more_display))
    mc3.metric("Tier 3", counts.get("tier3", 0))
    mc4.metric("Stay Away", len(stay_away_display))

    # Sanity-check warnings (shown when debug env is on)
    if _debug_env:
        for w in slate.get("debug_stats", {}).get("warnings", []):
            st.warning(f"Slate Debug: {w}")

    # Debug counters (behind toggle)
    debug = slate.get("debug")
    ds = slate.get("debug_stats")
    if debug and ds:
        with st.expander("Debug: Classification Breakdown", expanded=False):
            dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
            dc1.metric("Total", debug["total_recs"])
            dc2.metric("Tier 1A", debug.get("tier1a_count", 0))
            dc3.metric("Tier 1B", debug.get("tier1b_count", 0))
            dc4.metric("Tier 2", debug["tier2_count"])
            dc5.metric("Tier 3", debug.get("tier3_count", 0))
            dc6.metric("Stay Away", debug["stay_away_count"])
            st.markdown("**By Confidence:** " + ", ".join(
                f"{k}: {v}" for k, v in sorted(debug["by_confidence"].items())
            ))
            st.markdown("**By Quality Tier:** " + ", ".join(
                f"{k}: {v}" for k, v in sorted(debug["by_quality_tier"].items())
            ))

            # Gate failure counts
            st.markdown("---")
            st.markdown("**Tier 1 gate failures**")
            t1g = ds["tier1_gate_failures"]
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("Conf != High", t1g["confidence_not_high"])
            gc2.metric("QT != Elite/Strong", t1g["quality_tier_not_elite_strong"])
            gc3.metric("< Dyn Floor", t1g["below_dynamic_floor"])
            gc4.metric("Edge <= 0", t1g["edge_not_positive"])

            st.markdown("**Tier 2 gate failures**")
            t2g = ds["tier2_gate_failures"]
            gc5, gc6, gc7, gc8 = st.columns(4)
            gc5.metric("Conf != H/M", t2g["confidence_not_high_medium"])
            gc6.metric("QT != E/S/M", t2g["quality_tier_not_elite_strong_moderate"])
            gc7.metric("< Edge Floor", t2g["below_edge_floor"])
            gc8.metric("Edge Z Low", t2g["edge_z_too_low"])

            # Sigma + dynamic floor stats
            st.markdown("---")
            def _fmt_stat(s):
                if s["min"] is None:
                    return "\u2014"
                return (
                    f"min={s['min']:.4f}  "
                    f"med={s['median']:.4f}  "
                    f"max={s['max']:.4f}"
                )
            st.markdown(
                f"**Sigma stats:** {_fmt_stat(ds['sigma_stats'])}"
            )
            if ds["robust_sigma_stats"]["min"] is not None:
                st.markdown(
                    f"**Robust sigma:** {_fmt_stat(ds['robust_sigma_stats'])}"
                )
            st.markdown(
                f"**Dynamic floor stats:** {_fmt_stat(ds['dynamic_floor_stats'])}"
            )

            # Warnings
            for w in ds.get("warnings", []):
                st.warning(w)

        # EV distribution stats (from volume_tuning)
        vt = slate.get("volume_tuning")
        if vt and vt.get("total", 0) > 0:
            with st.expander("Debug: EV Distribution", expanded=False):
                def _fmt_pctiles(d):
                    if d.get("p50") is None:
                        return "\u2014"
                    return (
                        f"p10={d['p10']:.2f}  "
                        f"p50={d['p50']:.2f}  "
                        f"p90={d['p90']:.2f}"
                    )
                _p = _fmt_pctiles
                st.markdown(f"**Total recs:** {vt['total']}")
                st.markdown(
                    f"**EV/$100:** {_p(vt.get('edge_pct', {}))}"
                )
                st.markdown(f"**edge_z:** {_p(vt.get('edge_z', {}))}")
                st.markdown(f"**sigma:** {_p(vt.get('sigma', {}))}")
                st.markdown(
                    f"**hold:** {_p(vt.get('hold_median', {}))}"
                )
                st.markdown(
                    f"**books:** {_p(vt.get('books_used', {}))}"
                )

    # EV Math Debug — top 3 from the combined candidate list
    _render_ev_debug_panel(top_plays + more_plays)

    st.divider()

    # ── Top Plays (Tier 1A, or 1A+1B in Standard mode) ───────────────
    _top_label = "Top Plays (Tier 1A)" if pro_mode else "Top Plays (Tier 1)"
    st.subheader(_top_label)
    if not top_display:
        st.info("No top plays today under current thresholds.")
        if pro_mode:
            st.caption(
                "Tier 1A requires **High** confidence, **Elite/Strong** "
                "quality, books >= 6, hold <= 6%, "
                "edge >= (2.5 + 1.0 \u00d7 \u03c3)."
            )
        else:
            st.caption(
                "Tier 1 requires **High/Medium** confidence, "
                "**Elite/Strong/Moderate** quality, "
                "edge >= dynamic floor."
            )
        # Show closest candidates when no top plays qualify
        if show_closest and closest:
            st.markdown("**Closest to qualifying:**")
            _render_slate_table(closest[:5])
    else:
        _render_slate_date_groups(top_display, show_cards=True)

    st.divider()

    # ── More Plays (Tier 2, or 1B+Tier 2 in Pro mode) ────────────────
    _more_label = "More Plays (Tier 1B + Tier 2)" if pro_mode else "More Plays (Tier 2)"
    st.subheader(_more_label)
    if not more_display:
        st.info("No additional plays meet current filters.")
        st.caption(
            "Try turning off Strict mode or lowering the edge floor."
        )
    else:
        _render_slate_table(more_display)

    st.divider()

    # ── Stay Away ─────────────────────────────────────────────────────
    if show_stay_away:
        st.subheader("Stay Away")
        if not stay_away_display:
            st.info("No games flagged to avoid (by current logic).")
        else:
            for entry in stay_away_display:
                _render_stay_away_entry(entry)


def _render_stay_away_entry(entry: dict) -> None:
    """Render a single Stay Away entry with risk score and reasons."""
    market_label = _slate_market_label(entry)
    ct = entry.get("commence_time")
    time_str = f" \u00b7 {_format_start_time(ct)}" if ct else ""
    score = entry.get("avoid_score", 0.0)
    with st.container(border=True):
        hcol, scol = st.columns([5, 1])
        with hcol:
            st.markdown(
                f"**{entry['event']}** \u2014 "
                f"{entry['selection']} ({market_label}){time_str}"
            )
        with scol:
            st.metric("Risk", f"{score:.0f}")
        for reason in entry["avoid_reasons"]:
            st.markdown(f"- :red[{reason}]")


def _render_why_tooltip(entry: dict) -> None:
    """Render a 'Why?' expander with transparency details for a slate entry."""
    with st.expander("Why?", expanded=False):
        dyn_floor = entry.get("dynamic_edge_floor")
        r_sigma = entry.get("robust_sigma", 0.0)
        ev_sigma = entry.get("ev_sigma", 0.0)
        lines = [
            f"- **Confidence:** {entry.get('confidence', 'N/A')}",
            f"- **Quality tier:** {entry.get('quality_tier', 'N/A')}",
            f"- **Quality score:** {entry.get('quality_score', 'N/A')}",
            f"- **EV/$100:** ${entry.get('edge_pct', 0.0):+.2f}",
            f"- **Edge Z:** {entry.get('edge_z', 0.0):+.2f}",
            f"- **n_eff:** {entry.get('n_eff', 0.0):.1f}",
            f"- **Sigma (prob):** {r_sigma:.4f} | "
            f"**Sigma (EV):** {ev_sigma:.4f}",
        ]
        if dyn_floor is not None:
            lines.append(
                f"- **Dynamic Tier 1 floor:** ${dyn_floor:.2f}"
            )
        hold = entry.get("market_hold_median")
        if hold is not None:
            lines.append(f"- **Market hold (median):** {hold:.2f}%")
        reasons = entry.get("avoid_reasons", [])
        if reasons:
            lines.append("- **Reasons:** " + "; ".join(reasons))
        st.markdown("\n".join(lines))


def _slate_market_label(entry: dict) -> str:
    """Format market + line for a slate entry."""
    market = entry["market"]
    line = entry.get("line")
    if market == "spread" and line is not None:
        return f"Spread ({line:+.1f})"
    if market == "total" and line is not None:
        return f"Total ({line:.1f})"
    return market.title()


def _render_ev_debug_panel(entries: list[dict]) -> None:
    """Show EV math breakdown for the top 3 entries.

    Guarded by ``SHOW_EV_DEBUG``.  Does not affect ranking/filtering.
    """
    if not SHOW_EV_DEBUG or not entries:
        return
    top3 = entries[:3]
    with st.expander("EV Math Debug (top 3)", expanded=False):
        for i, e in enumerate(top3, 1):
            odds = e.get("best_odds", 0)
            d_best = _slip_a2d(odds) if odds else 0.0
            p = e.get("consensus_prob", 0.0)
            p_be = e.get("p_be", 0.0)
            edge_pp = e.get("edge_pp", 0.0)
            edge_pct_pp = e.get("edge_pct_pp", 0.0)
            ev_100 = e.get("ev_100", 0.0)
            books = e.get("books_used", 0)
            st.markdown(
                f"**#{i} {e.get('selection', '?')}** "
                f"({e.get('market', '?')}) "
                f"@ {e.get('best_sportsbook', '?')}"
            )
            st.text(
                f"  p = {p:.4f}   books = {books}\n"
                f"  d_best = {d_best:.4f}\n"
                f"  p_be = {p_be:.4f}\n"
                f"  edge_pp = {edge_pp:.4f}\n"
                f"  edge_pct = {edge_pct_pp:.2f}%\n"
                f"  ev_100 = ${ev_100:.2f}\n"
            )


def _render_ev_debug_shopping(
    standouts: list[dict],
) -> None:
    """Show EV debug for top 3 standouts on Best Lines page."""
    if not SHOW_EV_DEBUG or not standouts:
        return
    top3 = standouts[:3]
    with st.expander("EV Math Debug (top 3)", expanded=False):
        for i, s in enumerate(top3, 1):
            p = s.get("consensus_prob", 0.0)
            d_best = s.get("d_best", 0.0)
            d_ref = s.get("d_ref", 0.0)
            p_be = 1.0 / d_best if d_best > 0 else 0.0
            edge_pp = p - p_be
            edge_pct = 100.0 * edge_pp
            ev_100 = 100.0 * (p * d_best - 1.0)
            exec_adv = s.get("exec_adv_100", 0.0)
            st.markdown(
                f"**#{i} {s.get('selection', '?')}** "
                f"({s.get('market', '?')}) "
                f"@ {s.get('sportsbook', '?')}"
            )
            st.text(
                f"  p = {p:.4f}\n"
                f"  d_best = {d_best:.4f}   "
                f"d_ref = {d_ref:.4f}\n"
                f"  p_be = {p_be:.4f}\n"
                f"  edge_pp = {edge_pp:.4f}\n"
                f"  edge_pct = {edge_pct:.2f}%\n"
                f"  ev_100 = ${ev_100:.2f}\n"
                f"  exec_adv_100 = ${exec_adv:.2f}\n"
            )


def _render_slate_date_groups(entries: list[dict], *, show_cards: bool) -> None:
    """Render slate entries grouped by commence date."""
    today = date.today()
    tomorrow = today + timedelta(days=1)

    # Check if any entry has commence_time
    has_dates = any(e.get("commence_time") is not None for e in entries)

    if not has_dates:
        # No grouping — render flat
        if show_cards:
            for entry in entries:
                _render_slate_card(entry)
        return

    # Group by local date
    groups: dict[date | None, list[dict]] = defaultdict(list)
    for entry in entries:
        ct = entry.get("commence_time")
        if ct is not None:
            local_date = ct.astimezone().date() if ct.tzinfo else ct.date()
        else:
            local_date = None
        groups[local_date].append(entry)

    dated_keys = sorted(k for k in groups if k is not None)
    ordered_keys: list[date | None] = list(dated_keys)
    if None in groups:
        ordered_keys.append(None)

    for group_date in ordered_keys:
        if group_date is None:
            label = "Time TBD"
        elif group_date == today:
            label = f"Today \u2014 {group_date.strftime('%a %b %-d')}"
        elif group_date == tomorrow:
            label = f"Tomorrow \u2014 {group_date.strftime('%a %b %-d')}"
        else:
            label = group_date.strftime("%a %b %-d")

        st.markdown(f"**{label}**")

        if show_cards:
            for entry in groups[group_date]:
                _render_slate_card(entry)


def _render_slate_card(entry: dict) -> None:
    """Render a single Tier-1 slate card with Add-to-slip."""
    slip_book = _get_slip_book()
    market_label = _slate_market_label(entry)
    ct = entry.get("commence_time")
    time_str = _format_start_time(ct) if ct else "TBD"

    with st.container(border=True):
        cols = st.columns([4, 2, 2, 1.5])
        with cols[0]:
            st.markdown(
                f"**{entry['event']}**  \n"
                f"{entry['selection']} ({market_label}) at "
                f"{format_american(entry['best_odds'])} on "
                f"**{entry['best_sportsbook']}**"
            )
            st.caption(f"Start: {time_str}")
        with cols[1]:
            st.metric("EV/$100", f"${entry['edge_pct']:+.2f}")
            ez = entry.get('edge_z', 0.0)
            st.caption(f"Z: {ez:+.2f} | Q: {entry['quality_score']}")
        with cols[2]:
            st.metric("Slate Score", f"{entry['slate_score']:.0f}")
            st.caption(f"Confidence: {entry['confidence']}")
            k_sugg = entry.get("kelly_suggested", 0)
            if k_sugg and k_sugg > 0:
                br = _bankroll()
                if br > 0:
                    st.caption(
                        f"Size: {k_sugg * 100:.1f}% "
                        f"({fmt_money(br * k_sugg)})"
                    )
                else:
                    st.caption(f"Size: {k_sugg * 100:.1f}%")
        with cols[3]:
            can_add = not slip_book or entry["best_sportsbook"] == slip_book
            safe_key = entry["event_id"].replace(" ", "_")
            if can_add:
                if st.button(
                    "\u2795 Slip",
                    key=f"slate_add_{safe_key}_{entry['market']}",
                    type="primary",
                ):
                    leg = {
                        "sport": _sport_name(),
                        "event_name": entry["event"],
                        "sportsbook": entry["best_sportsbook"],
                        "market": BET_TYPE_SHORT.get(entry["market"], entry["market"]),
                        "selection": entry["selection"],
                        "line": entry.get("line"),
                        "odds": entry["best_odds"],
                        "fetched_at": "",
                    }
                    _try_add_leg(leg)
            elif slip_book:
                st.caption(f"Locked to {slip_book}")

            if st.button("View", key=f"slate_view_{safe_key}_{entry['market']}"):
                st.session_state["page"] = "detail"
                st.session_state["selected_game"] = entry["event"]
                st.rerun()
        _render_why_tooltip(entry)


def _render_slate_table(entries: list[dict]) -> None:
    """Render Tier-2 slate entries as a table with per-row Why? expanders."""
    rows = []
    for entry in entries:
        ct = entry.get("commence_time")
        time_str = _format_start_time(ct) if ct else "TBD"
        market_label = _slate_market_label(entry)
        k_sugg = entry.get("kelly_suggested", 0)
        br = _bankroll()
        if k_sugg and k_sugg > 0 and br > 0:
            sizing_str = f"{k_sugg * 100:.1f}% ({fmt_money(br * k_sugg)})"
        elif k_sugg and k_sugg > 0:
            sizing_str = f"{k_sugg * 100:.1f}%"
        else:
            sizing_str = ""
        rows.append({
            "Event": entry["event"],
            "Start": time_str,
            "Pick": f"{entry['selection']} ({market_label})",
            "Odds": format_american(entry["best_odds"]),
            "Book": entry["best_sportsbook"],
            "EV/$100": f"${entry['edge_pct']:+.2f}",
            "Edge Z": f"{entry.get('edge_z', 0.0):+.2f}",
            "Quality": entry["quality_score"],
            "Confidence": entry["confidence"],
            "Sizing": sizing_str,
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
    # Per-row Why? expanders below the table
    for entry in entries:
        _render_why_tooltip(entry)


# ---------------------------------------------------------------------------
# Bet History dialog
# ---------------------------------------------------------------------------

@st.dialog("Bet History", width="large")
def _bet_history_dialog():
    """Modal showing Active and Settled bets (loaded from SQLite)."""
    init_bet_state(st.session_state)
    try:
        with LineStore() as _db:
            active = load_bets_from_db(_db, status="active")
            settled = load_bets_from_db(_db, status=None)
        settled = [b for b in settled if b.status in ("won", "lost", "push")]
    except Exception:
        active = st.session_state["active_bets"]
        settled = st.session_state["settled_bets"]

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

                def _settle(bid, outcome, _key=""):
                    try:
                        with LineStore() as _s:
                            close_bet_clv(bid, _s)
                            settle_bet_persistent(bid, outcome, _s)
                    except Exception:
                        pass  # CLV close + DB settle is best-effort
                    settle_bet(st.session_state, bid, outcome)
                    st.session_state["_reopen_history"] = True
                    st.rerun()

                with scols[0]:
                    if st.button(
                        "Won", key=f"hist_won_{bet.id}",
                        type="primary", use_container_width=True,
                    ):
                        _settle(bet.id, "won")
                with scols[1]:
                    if st.button(
                        "Lost", key=f"hist_lost_{bet.id}",
                        use_container_width=True,
                    ):
                        _settle(bet.id, "lost")
                with scols[2]:
                    if st.button(
                        "Push", key=f"hist_push_{bet.id}",
                        use_container_width=True,
                    ):
                        _settle(bet.id, "push")

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
                            st.metric(
                                "Profit",
                                f":green[+{fmt_money(pnl)}]",
                            )
                        elif bet.status == "push":
                            st.metric("Profit", fmt_money(0))
                        else:
                            st.metric(
                                "Profit",
                                f":red[-{fmt_money(bet.stake)}]",
                            )
                        if bet.settled_at:
                            st.caption(
                                f"Settled {bet.settled_at[:16]}",
                            )

                    # -- CLV metrics (best-effort) --
                    try:
                        with LineStore() as _s:
                            clv_rows = _s.get_clv(bet.id)
                    except Exception:
                        clv_rows = []
                    if clv_rows:
                        parts: list[str] = []
                        for cr in clv_rows:
                            m = compute_clv(cr)
                            if m is None:
                                continue
                            cd = m["clv_decimal"]
                            cp = m["clv_prob"]
                            color = clv_color(cp)
                            parts.append(
                                f"Leg {cr['leg_index']+1}: "
                                f":{color}[CLV "
                                f"{cd:+.3f} dec "
                                f"/ {cp*100:+.2f}pp]"
                            )
                        if parts:
                            st.caption(" | ".join(parts))

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
# Performance page
# ---------------------------------------------------------------------------

_BREAKDOWN_LABELS: dict[str, str] = {
    "confidence_at_pick": "Confidence",
    "quality_tier_at_pick": "Quality Tier",
    "market": "Market Type",
    "sport": "Sport",
    "pick_sportsbook": "Sportsbook",
}


def _page_performance():
    st.title("Performance")
    st.caption("Closing Line Value (CLV) analytics across all settled legs.")

    store = LineStore(DB_PATH)
    rows = store.get_all_clv()

    if not rows:
        st.info("No settled legs with CLV data yet. Settle some bets first!")
        return

    df_full = build_clv_dataframe(rows)
    if df_full.empty:
        st.info("No CLV data available.")
        return

    # ---- Filters ----
    with st.expander("Filters", expanded=False):
        fcols = st.columns(5)
        with fcols[0]:
            date_start = st.date_input("From", value=None, key="perf_date_start")
        with fcols[1]:
            date_end = st.date_input("To", value=None, key="perf_date_end")
        with fcols[2]:
            sports = ["All"] + sorted(
                df_full["sport"].dropna().unique().tolist()
            )
            sport_filter = st.selectbox("Sport", sports, key="perf_sport")
        with fcols[3]:
            markets = ["All"] + sorted(
                df_full["market"].dropna().unique().tolist()
            )
            market_filter = st.selectbox("Market", markets, key="perf_market")
        with fcols[4]:
            confs = ["All"] + sorted(
                df_full["confidence_at_pick"].dropna().unique().tolist()
            )
            conf_filter = st.selectbox(
                "Confidence", confs, key="perf_confidence",
            )

        tiers = ["All"] + sorted(
            df_full["quality_tier_at_pick"].dropna().unique().tolist()
        )
        tier_filter = st.selectbox(
            "Quality Tier", tiers, key="perf_tier",
        )

    df = apply_filters(
        df_full,
        date_start=str(date_start) if date_start else None,
        date_end=str(date_end) if date_end else None,
        sport=sport_filter if sport_filter != "All" else None,
        market=market_filter if market_filter != "All" else None,
        confidence=conf_filter if conf_filter != "All" else None,
        quality_tier=tier_filter if tier_filter != "All" else None,
    )

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    # ---- KPI cards ----
    kpis = summary_kpis(df)

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Legs (Closed)", kpis["total_legs"])
    k2.metric("Beating Close %", f"{kpis['beating_close_pct']}%")
    k3.metric("Avg CLV (prob pts)", f"{kpis['avg_clv_prob']:+.4f}")

    k4, k5, k6 = st.columns(3)
    k4.metric("Median CLV (prob pts)", f"{kpis['median_clv_prob']:+.4f}")
    k5.metric("Avg CLV (decimal)", f"{kpis['avg_clv_decimal']:+.4f}")
    k6.metric("Median CLV (decimal)", f"{kpis['median_clv_decimal']:+.4f}")

    st.divider()

    # ---- Breakdown tables ----
    st.subheader("Breakdowns")
    breakdowns = all_breakdowns(df)

    if breakdowns:
        tabs = st.tabs([
            _BREAKDOWN_LABELS.get(col, col) for col in breakdowns
        ])
        for tab, (col, tbl) in zip(tabs, breakdowns.items()):
            with tab:
                display_tbl = tbl.rename(columns={
                    col: _BREAKDOWN_LABELS.get(col, col),
                    "legs": "Legs",
                    "beating_pct": "Beating %",
                    "avg_clv_decimal": "Avg CLV (dec)",
                    "avg_clv_prob": "Avg CLV (prob)",
                })
                st.dataframe(
                    display_tbl,
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Not enough metadata for breakdowns.")

    st.divider()

    # ---- Charts ----
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("30-Day Rolling CLV")
        rolling = rolling_clv_series(df)
        if not rolling.empty:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(
                rolling["date"],
                rolling["rolling_clv_prob"],
                linewidth=1.5,
            )
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylabel("Rolling CLV (prob)")
            ax.set_xlabel("")
            fig.autofmt_xdate(rotation=30)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Not enough data for rolling chart.")

    with chart_right:
        st.subheader("CLV Distribution")
        dist = clv_distribution(df)
        if not dist.empty:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3))
            colors = [
                "#d9534f" if "< " in lbl or lbl.startswith("[-")
                else "#5cb85c"
                for lbl in dist["bin_label"]
            ]
            ax.bar(
                range(len(dist)),
                dist["count"],
                color=colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(range(len(dist)))
            ax.set_xticklabels(dist["bin_label"], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Count")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Not enough data for distribution chart.")

    # ---- Calibration Panel ------------------------------------------------
    st.divider()
    st.subheader("Tier Calibration")
    st.caption(
        "Compares CLV performance across proxy tier groups "
        "(Tier 1 / Tier 2 / Stay Away based on pick-time metadata)."
    )

    cal = calibration_stats(df)
    if cal:
        cc1, cc2, cc3 = st.columns(3)
        for col, tier in zip((cc1, cc2, cc3), ("Tier 1", "Tier 2", "Stay Away")):
            stats = cal.get(tier, {})
            with col:
                with st.container(border=True):
                    st.markdown(f"**{tier}**")
                    st.metric("Legs", stats.get("legs", 0))
                    st.metric(
                        "Beating Close %",
                        f"{stats.get('beating_pct', 0.0)}%",
                    )
                    st.metric(
                        "Avg CLV (prob)",
                        f"{stats.get('avg_clv_prob', 0.0):+.4f}",
                    )

        # Advisory message when Tier 1 does not outperform Tier 2
        t1 = cal.get("Tier 1", {})
        t2 = cal.get("Tier 2", {})
        if (
            t1.get("legs", 0) >= 5
            and t2.get("legs", 0) >= 5
            and t1.get("avg_clv_prob", 0.0) <= t2.get("avg_clv_prob", 0.0)
        ):
            st.warning(
                "Tier 1 is not outperforming Tier 2 on average CLV. "
                "Consider enabling **Pro Mode** on the Daily Slate to "
                "tighten Tier 1 criteria, or review your edge thresholds."
            )
    else:
        st.info(
            "Not enough pick-time metadata for calibration "
            "(needs confidence, quality tier, and edge at pick)."
        )

    # ---- Auto-Calibrate Button -------------------------------------------
    st.divider()
    st.subheader("Auto-Calibrate Thresholds")
    st.caption(
        "Derive tier thresholds from historical CLV. "
        "Results are saved and used by **Auto** mode on the Daily Slate."
    )
    confirm_cal = st.checkbox(
        "I understand this will update saved thresholds",
        key="perf_cal_confirm",
    )
    if st.button(
        "Calibrate",
        key="perf_cal_btn",
        disabled=not confirm_cal,
    ):
        train_df = load_clv_training_df(store)
        result = calibrate_thresholds(train_df)
        store.save_calibration("global", calibration_to_json(result))
        st.success("Calibration saved (global).")
        report = format_calibration_report(result)
        st.code(report, language="text")

    # Show current calibration if it exists
    existing = store.load_calibration("global")
    if existing:
        with st.expander("Current saved calibration", expanded=False):
            ex_dict = calibration_from_json(existing)
            st.code(
                format_calibration_report(ex_dict), language="text",
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
        elif nav == "Daily Slate":
            _page_daily_slate()
        elif nav == "Performance":
            _page_performance()
        else:
            _page_dashboard()


if __name__ == "__main__":
    main()

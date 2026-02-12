"""Streamlit web dashboard — run with: streamlit run src/line_tracker/dashboard.py"""

from __future__ import annotations

import os
from datetime import datetime

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

        with LineStore(DB_PATH) as store:
            count = store.save_lines(lines)

        st.success(f"Got {len(lines)} lines across {sport}. Saved {count} new rows.")
        st.session_state["last_fetch"] = lines

    lines = st.session_state.get("last_fetch")
    if not lines:
        return

    compact = _compact_mode()
    max_rows = _max_display_rows()
    display_df = _lines_to_df(lines, compact=compact)
    raw_df = _lines_to_raw_df(lines)

    # Summary metrics
    books = display_df["Sportsbook"].nunique()
    games = display_df["Game"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Sportsbooks", books)
    c2.metric("Games", games)
    c3.metric("Total Lines", len(display_df))

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        events = ["All games"] + sorted(display_df["Game"].unique().tolist())
        event_pick = st.selectbox("Game", events, key="live_game")
    with col2:
        type_opts = ["All types"] + list(
            BET_TYPE_SHORT.values() if compact else BET_TYPE_LABELS.values()
        )
        type_pick = st.selectbox("Bet type", type_opts, key="live_type")
    with col3:
        book_opts = ["All books"] + sorted(
            display_df["Sportsbook"].unique().tolist()
        )
        book_pick = st.selectbox("Sportsbook", book_opts, key="live_book")

    mask = pd.Series(True, index=display_df.index)
    if event_pick != "All games":
        mask &= display_df["Game"] == event_pick
    if type_pick != "All types":
        mask &= display_df["Bet Type"] == type_pick
    if book_pick != "All books":
        mask &= display_df["Sportsbook"] == book_pick

    filtered_display = display_df[mask].head(max_rows)
    filtered_raw = raw_df[mask].head(max_rows)

    if not filtered_display.empty and len(filtered_display) <= max_rows:
        styled = _style_best_lines(filtered_display, filtered_raw)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(filtered_display, use_container_width=True, hide_index=True)

    total_matching = int(mask.sum())
    if total_matching > max_rows:
        st.caption(
            f"Showing {max_rows} of {total_matching} rows (adjust in sidebar)"
        )


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
        st.info(
            "No data yet. Head to the **Live Odds** tab and fetch some odds first."
        )
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

    compact = _compact_mode()
    display_df = _lines_to_df(latest, compact=compact)
    raw_df = _lines_to_raw_df(latest)

    if not display_df.empty:
        styled = _style_best_lines(display_df, raw_df)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Best-value callouts
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


# ---------------------------------------------------------------------------
# Tab: Arbitrage
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

    for alert in alerts:
        if alert.level.value == "critical":
            st.error(alert.message)

    # Profitable arbs — expanded cards with stake calculator
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

                # Actionable stake calculator for moneyline arbs
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
                                    f"**Leg A — {arb.side_a.sportsbook}**  \n"
                                    f"Bet Home at "
                                    f"{_format_odds(arb.side_a.home_value)}  \n"
                                    f"Stake: **${stake_a:.2f}**"
                                )
                            with lc2:
                                st.markdown(
                                    f"**Leg B — {arb.side_b.sportsbook}**  \n"
                                    f"Bet Away at "
                                    f"{_format_odds(arb.side_b.away_value)}  \n"
                                    f"Stake: **${stake_b:.2f}**"
                                )
                            st.divider()
                            pc1, pc2, pc3 = st.columns(3)
                            pc1.metric("Total wagered", "$100.00")
                            pc2.metric("Guaranteed profit", f"${profit:.2f}")
                            pc3.metric("ROI", f"{roi:.2f}%")

    # Near arbs table
    if near:
        with st.expander(
            f"Near-arbs ({len(near)}) — not profitable yet, but close",
        ):
            rows = []
            for arb in near:
                rows.append({
                    "Game": arb.event,
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


# ---------------------------------------------------------------------------
# Tab: Line Movements
# ---------------------------------------------------------------------------

_TIME_WINDOWS = {
    "All": 0,
    "Last 15m": 15,
    "Last 1h": 60,
    "Last 6h": 360,
    "Last 24h": 1440,
}


def _tab_movements():
    st.subheader("Line Movements")
    st.caption(
        "Compares your older fetches with newer ones to spot which lines moved. "
        "Big moves can signal sharp action or breaking news."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        threshold = st.slider(
            "Minimum movement size",
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
        time_window = st.selectbox(
            "Time window",
            list(_TIME_WINDOWS.keys()),
            key="move_time_window",
            help="Filter by how recently the new line was fetched.",
        )

    # "Juice-only" approximation: if the primary value barely changed,
    # the movement is likely just a juice adjustment. We hide moves where
    # abs(change) < 1 for moneyline or < 0.5 for spreads/totals.
    hide_juice = st.toggle(
        "Hide juice-only changes",
        value=False,
        key="hide_juice",
        help=(
            "Hides moves where the primary line barely changed "
            "(< 1 for moneyline, < 0.5 for spreads/totals), "
            "which are likely juice adjustments only."
        ),
    )

    detect = st.button("Detect Movements", type="primary", key="btn_moves")

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

    # ---- post-filters applied to already-detected moves ----
    filtered = list(moves)

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

    # Juice-only filter
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

    # Alerts
    for alert in alerts:
        if alert.level.value == "critical":
            st.error(alert.message)
        elif alert.level.value == "warning":
            st.warning(alert.message)

    # Summary
    up = sum(1 for m in filtered if m.direction == "up")
    down = sum(1 for m in filtered if m.direction == "down")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Moves", len(filtered))
    c2.metric("Moved Up", up)
    c3.metric("Moved Down", down)

    # Build table
    max_rows = _max_display_rows()
    display_slice = filtered[:max_rows]

    rows = []
    significant_flags: list[bool] = []
    for m in display_slice:
        arrow = "\u2191" if m.direction == "up" else (
            "\u2193" if m.direction == "down" else "\u2014"
        )
        old_str = _format_value(m.old_line.home_value, m.bet_type)
        new_str = _format_value(m.new_line.home_value, m.bet_type)

        # Percent change (skip when baseline is zero to avoid div-by-zero)
        if m.old_line.home_value != 0:
            pct = (m.change / abs(m.old_line.home_value)) * 100
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = "\u2014"

        rows.append({
            "Sportsbook": m.sportsbook,
            "Game": m.event,
            "Bet Type": BET_TYPE_LABELS.get(m.bet_type.value, m.bet_type.value),
            "Old": old_str,
            "": "\u2192",
            "New": new_str,
            "Delta": f"{arrow} {m.change:+.1f}",
            "% Change": pct_str,
            "When": _relative_time(m.new_line.timestamp),
        })

        # Mark "significant" if the move is large for its type
        if m.bet_type == BetType.MONEYLINE:
            significant_flags.append(abs(m.change) >= 10)
        else:
            significant_flags.append(abs(m.change) >= 1.0)

    move_df = pd.DataFrame(rows)

    # Apply subtle highlight to significant-movement rows
    if not move_df.empty and len(move_df) <= max_rows:
        sig = significant_flags  # captured from loop above

        def _apply_move_styles(df: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, is_sig in enumerate(sig):
                if is_sig and i < len(styles):
                    styles.iloc[i] = "background-color: #fff3cd"
            return styles

        styled = move_df.style.apply(
            lambda x: _apply_move_styles(move_df), axis=None,
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(move_df, use_container_width=True, hide_index=True)

    if len(filtered) > max_rows:
        st.caption(
            f"Showing {max_rows} of {len(filtered)} movements "
            "(adjust in sidebar)"
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
            "Game", ["All games"] + events, key="hist_event",
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

    compact = _compact_mode()
    df = _lines_to_df(lines, compact=compact)
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

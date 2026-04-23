"""MLB Player Props page — HR, Total Bases, Pitcher Ks."""

from __future__ import annotations

import logging
import os

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


def _pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def _fmt(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _edge_color(edge: float) -> str:
    if edge >= 0.05:
        return "🟢"
    if edge >= 0.02:
        return "🟡"
    return "🔴"


def _american_to_prob(price: int | None) -> float | None:
    """Convert American odds to implied probability."""
    if price is None:
        return None
    try:
        p = int(price)
    except (TypeError, ValueError):
        return None
    if p > 0:
        return 100.0 / (p + 100.0)
    return abs(p) / (abs(p) + 100.0)


def _render_prop_lines_section(
    pred: dict,
    prop_lines: dict,
    api_key: str,
) -> None:
    """Render edge table for a single game's prop lines."""
    from line_tracker.model.mlb_data import match_prop_to_prediction

    st.subheader("📊 Prop Line Edge")

    hr_props = pred.get("hr_props") or []
    batter_names = [b.get("batter_name", "") for b in hr_props if b.get("batter_name")]

    rows = []
    for event_id, market_data in prop_lines.items():
        for market, player_list in market_data.items():
            market_label = {
                "batter_home_runs": "HR",
                "batter_total_bases": "TB",
                "pitcher_strikeouts": "K",
            }.get(market, market)

            for entry in player_list or []:
                player = entry.get("player", "")
                line = entry.get("line")
                over_price = entry.get("over_price")
                under_price = entry.get("under_price")

                implied_prob = _american_to_prob(over_price)

                # Try to find matching model prediction
                model_prob: float | None = None
                if market == "batter_home_runs":
                    matched = match_prop_to_prediction(player, batter_names)
                    if matched:
                        for b in hr_props:
                            if b.get("batter_name") == matched:
                                model_prob = b.get("hr_prob")
                                break
                elif market == "pitcher_strikeouts":
                    for side in ("home_k_props", "away_k_props"):
                        k_props = pred.get(side) or {}
                        pname = k_props.get("pitcher_name", "")
                        if pname and match_prop_to_prediction(player, [pname]):
                            lines_dict = k_props.get("prop_lines") or {}
                            # Use nearest line
                            if line is not None:
                                key = f"over_{line}"
                                model_prob = lines_dict.get(key) or lines_dict.get("over_4.5")
                            break

                edge = None
                edge_str = "—"
                badge = ""
                if model_prob is not None and implied_prob is not None:
                    edge = model_prob - implied_prob
                    edge_str = f"{edge * 100:+.1f}%"
                    badge = _edge_color(edge)

                rows.append({
                    "Market":       market_label,
                    "Player":       player,
                    "Line":         f"O{line}" if line else "—",
                    "Over":         str(over_price) if over_price is not None else "—",
                    "Under":        str(under_price) if under_price is not None else "—",
                    "Implied%":     _pct(implied_prob),
                    "Model%":       _pct(model_prob),
                    "Edge":         f"{badge} {edge_str}".strip(),
                })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No prop lines fetched yet.")


def render_mlb_props_page() -> None:
    st.header("⚾ MLB Player Props")
    st.caption("Predictions derived from MLB Stats API season data — no prop lines required.")

    with st.spinner("Loading predictions…"):
        try:
            from line_tracker.model.mlb_predict import predict_mlb_games
            preds = predict_mlb_games()
        except Exception as exc:
            st.error(f"Could not load predictions: {exc}")
            return

    if not preds:
        st.info("No games found for today.")
        return

    games_with_props = [
        p for p in preds
        if p.get("hr_props") or p.get("home_k_props") or p.get("away_k_props")
    ]

    if not games_with_props:
        st.info("No prop data available yet — lineups may not be posted.")
        # Still render all games to show K props from pitching stats
        games_with_props = preds

    # ── Odds API prop line fetch controls ───────────────────────────────────
    st.divider()
    with st.expander("⚡ Fetch Prop Lines from Odds API", expanded=False):
        api_key = st.text_input(
            "Odds API Key",
            value=os.environ.get("ODDS_API_KEY", ""),
            type="password",
            help="Get a free key at the-odds-api.com",
        )
        markets_sel = st.multiselect(
            "Markets to fetch",
            ["batter_home_runs", "batter_total_bases", "pitcher_strikeouts"],
            default=["batter_home_runs", "pitcher_strikeouts"],
        )
        force_refresh = st.checkbox("Force refresh cache", value=False)
        fetch_btn = st.button("Fetch Prop Lines", type="primary")

        fetched_prop_lines: dict = st.session_state.get("prop_lines_cache", {})

        if fetch_btn:
            if not api_key:
                st.warning("Enter an Odds API key to fetch prop lines.")
            elif not markets_sel:
                st.warning("Select at least one market.")
            else:
                with st.spinner("Fetching event IDs…"):
                    try:
                        from line_tracker.model.mlb_data import (
                            fetch_today_mlb_event_ids,
                            fetch_prop_lines,
                        )
                        event_ids = fetch_today_mlb_event_ids(api_key)
                        if not event_ids:
                            st.warning("No MLB events found for today.")
                        else:
                            st.info(f"Found {len(event_ids)} event(s). Fetching props…")
                            with st.spinner("Fetching prop lines…"):
                                result = fetch_prop_lines(
                                    event_ids=event_ids,
                                    markets=markets_sel,
                                    api_key=api_key,
                                    force_refresh=force_refresh,
                                )
                            st.session_state["prop_lines_cache"] = result
                            fetched_prop_lines = result
                            total = sum(
                                len(mdata)
                                for edata in result.values()
                                for mdata in edata.values()
                            )
                            st.success(f"Fetched {total} player prop entries across {len(result)} event(s).")
                    except Exception as exc:
                        st.error(f"Prop fetch failed: {exc}")

    # ── Per-game sections ────────────────────────────────────────────────────
    for pred in games_with_props:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        home_p = pred.get("home_pitcher") or "TBD"
        away_p = pred.get("away_pitcher") or "TBD"

        with st.expander(f"**{away} @ {home}**  ·  {home_p} vs {away_p}", expanded=False):

            # ── HR Props ─────────────────────────────────────────────────
            hr_props = pred.get("hr_props") or []
            if hr_props:
                st.subheader("💥 HR Props")
                hr_rows = []
                for b in hr_props:
                    sim_badge = " ⚡" if b.get("sim_used") else ""
                    hr_rows.append({
                        "Batter":        b.get("batter_name", "") + sim_badge,
                        "Team":          b.get("team", ""),
                        "Order":         b.get("batting_order", ""),
                        "HR Prob":       _pct(b.get("hr_prob")),
                        "Exp HRs":       _fmt(b.get("expected_hrs"), 3),
                        "AB/HR":         _fmt(b.get("ab_per_hr"), 1),
                        "Pitcher Adj":   _fmt(b.get("vs_pitcher_adj"), 3),
                        "Method":        "Sim-based ⚡" if b.get("sim_used") else "Model-based",
                    })
                if hr_rows:
                    st.dataframe(pd.DataFrame(hr_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("HR props unavailable — lineup not yet posted.")

            # ── Total Bases Props ─────────────────────────────────────────
            home_tb = pred.get("home_tb_props") or []
            away_tb = pred.get("away_tb_props") or []
            if home_tb or away_tb:
                st.subheader("🏃 Total Bases Props")
                tb_rows = []
                for b in (away_tb[:5] + home_tb[:5]):
                    tb_rows.append({
                        "Batter":    b.get("batter_name", ""),
                        "Team":      b.get("team", ""),
                        "Order":     b.get("batting_order", ""),
                        "Exp TB":    _fmt(b.get("expected_tbs"), 2),
                        "Over 1.5":  _pct(b.get("over_1_5")),
                        "Over 2.5":  _pct(b.get("over_2_5")),
                    })
                if tb_rows:
                    st.dataframe(pd.DataFrame(tb_rows), use_container_width=True, hide_index=True)

            # ── Pitcher K Props ───────────────────────────────────────────
            st.subheader("🎯 Pitcher K Props")
            k_cols = st.columns(2)
            for col, side, k_props in [
                (k_cols[0], f"{home_p} ({home})", pred.get("home_k_props") or {}),
                (k_cols[1], f"{away_p} ({away})", pred.get("away_k_props") or {}),
            ]:
                with col:
                    st.markdown(f"**{side}**")
                    if not k_props or not k_props.get("expected_ks"):
                        st.caption("No data")
                        continue
                    lines = k_props.get("prop_lines") or {}
                    k_data = {
                        "Metric": [
                            "Expected Ks",
                            "K prob / PA",
                            "Exp innings",
                            "Opp lineup K%",
                            "Over 4.5",
                            "Over 5.5",
                            "Over 6.5",
                            "Over 7.5",
                        ],
                        "Value": [
                            _fmt(k_props.get("expected_ks")),
                            _pct(k_props.get("k_prob_per_pa")),
                            _fmt(k_props.get("expected_innings"), 1),
                            _pct(k_props.get("vs_lineup_k_rate")),
                            _pct(lines.get("over_4.5")),
                            _pct(lines.get("over_5.5")),
                            _pct(lines.get("over_6.5")),
                            _pct(lines.get("over_7.5")),
                        ],
                    }
                    st.dataframe(
                        pd.DataFrame(k_data),
                        use_container_width=True,
                        hide_index=True,
                    )

            # ── Prop Line Edge (if lines fetched) ─────────────────────────
            if fetched_prop_lines:
                _render_prop_lines_section(pred, fetched_prop_lines, api_key)

            # ── Simulator Section ─────────────────────────────────────────
            st.subheader("🎯 Simulator")
            st.caption("Monte Carlo PA simulator using Statcast outcome distributions.")

            batter_options: list[tuple[str, int]] = []
            for b in hr_props or []:
                bname = b.get("batter_name", "")
                bid_raw = b.get("batter_id")
                if bname and bid_raw:
                    batter_options.append((bname, int(bid_raw)))

            # Fallback: build name→id from batter_stats via session state
            if not batter_options:
                st.caption("Lineup batter IDs not available — enter IDs manually below.")
                col_b, col_p = st.columns(2)
                with col_b:
                    batter_id_input = st.number_input(
                        "Batter MLBAM ID",
                        min_value=1,
                        value=592450,
                        step=1,
                        key=f"sim_bid_{home}_{away}",
                    )
                with col_p:
                    pitcher_id_input = st.number_input(
                        "Pitcher MLBAM ID",
                        min_value=1,
                        value=571466,
                        step=1,
                        key=f"sim_pid_{home}_{away}",
                    )
                batter_id_sel = int(batter_id_input)
                pitcher_id_sel = int(pitcher_id_input)
                batter_label = f"Batter #{batter_id_sel}"
                pitcher_label = f"Pitcher #{pitcher_id_sel}"
            else:
                batter_labels = [f"{name}" for name, _ in batter_options]
                batter_sel_idx = st.selectbox(
                    "Batter",
                    range(len(batter_labels)),
                    format_func=lambda i: batter_labels[i],
                    key=f"sim_batter_{home}_{away}",
                )
                batter_id_sel = batter_options[batter_sel_idx][1]
                batter_label = batter_labels[batter_sel_idx]

                pitcher_options: list[tuple[str, int]] = []
                for side_key, pid_key, pname_key in [
                    ("home_k_props", "home_pitcher_id", "home_pitcher"),
                    ("away_k_props", "away_pitcher_id", "away_pitcher"),
                ]:
                    pid = pred.get(pid_key)
                    pname = pred.get(pname_key) or ""
                    if pid and pname:
                        pitcher_options.append((pname, int(pid)))

                if pitcher_options:
                    p_labels = [f"{name}" for name, _ in pitcher_options]
                    p_sel_idx = st.selectbox(
                        "Pitcher",
                        range(len(p_labels)),
                        format_func=lambda i: p_labels[i],
                        key=f"sim_pitcher_{home}_{away}",
                    )
                    pitcher_id_sel = pitcher_options[p_sel_idx][1]
                    pitcher_label = p_labels[p_sel_idx]
                else:
                    pitcher_id_sel = st.number_input(
                        "Pitcher MLBAM ID",
                        min_value=1,
                        value=571466,
                        step=1,
                        key=f"sim_pid2_{home}_{away}",
                    )
                    pitcher_label = f"Pitcher #{pitcher_id_sel}"

            sim_col1, sim_col2 = st.columns([2, 1])
            with sim_col1:
                n_sims = st.slider(
                    "Simulations",
                    min_value=1000,
                    max_value=10000,
                    value=5000,
                    step=1000,
                    key=f"sim_n_{home}_{away}",
                )
            with sim_col2:
                exp_pas = st.number_input(
                    "Expected PAs",
                    min_value=1.0,
                    max_value=6.0,
                    value=3.8,
                    step=0.1,
                    key=f"sim_pas_{home}_{away}",
                )

            run_sim_btn = st.button(
                "▶ Run Simulation",
                key=f"run_sim_{home}_{away}",
            )

            if run_sim_btn:
                with st.spinner(f"Simulating {n_sims:,} PAs…"):
                    try:
                        from line_tracker.model.mlb_simulator import simulate_game_props
                        sim_result = simulate_game_props(
                            batter_id=batter_id_sel,
                            pitcher_id=pitcher_id_sel,
                            expected_pas=float(exp_pas),
                            n_simulations=n_sims,
                        )
                        method_badge = "Sim-based ⚡" if (
                            sim_result.get("used_statcast_batter") or sim_result.get("used_statcast_pitcher")
                        ) else "League avg fallback"

                        st.info(f"Method: **{method_badge}**  |  {n_sims:,} simulations")

                        sim_rows = [
                            {"Prop": "HR (≥1)",        "Probability": _pct(sim_result["hr_prob"])},
                            {"Prop": "TB Over 1.5",    "Probability": _pct(sim_result["over_1_5_tb"])},
                            {"Prop": "TB Over 2.5",    "Probability": _pct(sim_result["over_2_5_tb"])},
                            {"Prop": "Hit (≥1)",       "Probability": _pct(sim_result["over_0_5_hits"])},
                            {"Prop": "K (≥1 batter)",  "Probability": _pct(sim_result["k_prob"])},
                            {"Prop": "Exp. TB",        "Probability": _fmt(sim_result["expected_tb"])},
                            {"Prop": "Exp. Hits",      "Probability": _fmt(sim_result["expected_hits"])},
                        ]
                        st.dataframe(
                            pd.DataFrame(sim_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    except Exception as exc:
                        st.error(f"Simulation failed: {exc}")

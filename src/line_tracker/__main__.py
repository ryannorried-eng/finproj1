"""CLI entry point: python -m line_tracker"""

from __future__ import annotations

import argparse
import sys

from line_tracker.alerts import AlertManager
from line_tracker.arbitrage import find_moneyline_arbs, find_spread_arbs
from line_tracker.models import BetType
from line_tracker.movements import detect_moves
from line_tracker.scraper import OddsClient
from line_tracker.storage import LineStore

SPORTS = {
    "nfl": "americanfootball_nfl",
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "ncaaf": "americanfootball_ncaaf",
    "ncaab": "basketball_ncaab",
    "mma": "mma_mixed_martial_arts",
    "soccer": "soccer_epl",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="line_tracker",
        description="Fetch, track, and analyze sports betting lines",
    )
    sub = parser.add_subparsers(dest="command")

    # --- fetch ---
    fetch_p = sub.add_parser("fetch", help="Fetch live odds and save")
    fetch_p.add_argument(
        "--sport",
        default="nfl",
        choices=list(SPORTS.keys()),
        help="Sport to fetch (default: nfl)",
    )
    fetch_p.add_argument("--db", default="lines.db", help="DB path")

    # --- lines ---
    lines_p = sub.add_parser("lines", help="Show stored lines")
    lines_p.add_argument("--event", help="Filter by event name")
    lines_p.add_argument(
        "--type",
        choices=["moneyline", "spread", "total"],
        help="Filter by bet type",
    )
    lines_p.add_argument(
        "--limit", type=int, default=50, help="Max rows"
    )
    lines_p.add_argument("--db", default="lines.db", help="DB path")

    # --- arbs ---
    arbs_p = sub.add_parser(
        "arbs", help="Scan for arbitrage opportunities"
    )
    arbs_p.add_argument(
        "--sport",
        default="nfl",
        choices=list(SPORTS.keys()),
    )
    arbs_p.add_argument("--db", default="lines.db", help="DB path")

    # --- moves ---
    moves_p = sub.add_parser(
        "moves", help="Detect line movements since last fetch"
    )
    moves_p.add_argument("--event", help="Filter by event")
    moves_p.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="Min change to show",
    )
    moves_p.add_argument("--db", default="lines.db", help="DB path")

    # --- events ---
    events_p = sub.add_parser("events", help="List tracked events")
    events_p.add_argument("--db", default="lines.db", help="DB path")

    # --- sports ---
    sub.add_parser("sports", help="List available sports from API")

    # --- slate ---
    slate_p = sub.add_parser("slate", help="Build and display daily slate")
    slate_p.add_argument(
        "--sport", default="nba", choices=list(SPORTS.keys()),
    )
    slate_p.add_argument("--db", default="lines.db", help="DB path")
    slate_p.add_argument(
        "--mode", default="Standard",
        choices=["Standard", "Pro", "Auto"],
    )

    # --- snapshot ---
    snap_p = sub.add_parser("snapshot", help="Log recommendation snapshots")
    snap_p.add_argument(
        "--sport", default="nba", choices=list(SPORTS.keys()),
    )
    snap_p.add_argument("--db", default="lines.db", help="DB path")

    # --- cycle ---
    cycle_p = sub.add_parser(
        "cycle", help="Run one automation cycle (slate+prune+snapshot)",
    )
    cycle_p.add_argument(
        "--sport", default="nba", choices=list(SPORTS.keys()),
    )
    cycle_p.add_argument("--db", default="lines.db", help="DB path")
    cycle_p.add_argument(
        "--mode", default="Standard",
        choices=["Standard", "Pro", "Auto"],
    )
    cycle_p.add_argument(
        "--top-n", type=int, default=3, help="Max top picks",
    )
    cycle_p.add_argument("--dry-run", action="store_true")

    # --- import-outcomes ---
    import_p = sub.add_parser(
        "import-outcomes", help="Import game outcomes from CSV",
    )
    import_p.add_argument("csv_file", help="Path to outcomes CSV")
    import_p.add_argument("--db", default="lines.db", help="DB path")

    # --- roi-report ---
    roi_p = sub.add_parser("roi-report", help="Show ROI report")
    roi_p.add_argument("--db", default="lines.db", help="DB path")

    # --- train-model ---
    train_p = sub.add_parser(
        "train-model", help="Train/refresh CLV prediction model",
    )
    train_p.add_argument("--db", default="lines.db", help="DB path")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "sports":
        return _cmd_sports()
    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "lines":
        return _cmd_lines(args)
    if args.command == "arbs":
        return _cmd_arbs(args)
    if args.command == "moves":
        return _cmd_moves(args)
    if args.command == "events":
        return _cmd_events(args)
    if args.command == "slate":
        return _cmd_slate(args)
    if args.command == "snapshot":
        return _cmd_snapshot(args)
    if args.command == "cycle":
        return _cmd_cycle(args)
    if args.command == "import-outcomes":
        return _cmd_import_outcomes(args)
    if args.command == "roi-report":
        return _cmd_roi_report(args)
    if args.command == "train-model":
        return _cmd_train_model(args)
    return 0


def _cmd_sports() -> int:
    with OddsClient() as client:
        sports = client.get_sports()
    print(f"{'Key':<40} {'Title':<30} {'Active'}")
    print("-" * 76)
    for s in sports:
        active = "yes" if s.get("active") else "no"
        print(f"{s['key']:<40} {s['title']:<30} {active}")
    return 0


def _cmd_fetch(args) -> int:
    sport_key = SPORTS[args.sport]
    print(f"Fetching {args.sport} odds...")

    with OddsClient() as client:
        lines = client.get_odds(sport=sport_key)

    if not lines:
        print("No lines available.")
        return 0

    with LineStore(args.db) as store:
        count = store.save_lines(lines)

    print(f"Saved {count} lines to {args.db}")
    _print_lines_table(lines[:20])
    return 0


def _cmd_lines(args) -> int:
    bt = BetType(args.type) if args.type else None
    with LineStore(args.db) as store:
        lines = store.get_lines(
            event=args.event, bet_type=bt, limit=args.limit
        )
    if not lines:
        print("No lines found.")
        return 0
    _print_lines_table(lines)
    return 0


def _cmd_arbs(args) -> int:
    sport_key = SPORTS[args.sport]
    print(f"Scanning {args.sport} for arbitrage...")

    with OddsClient() as client:
        lines = client.get_odds(sport=sport_key)

    ml_arbs = find_moneyline_arbs(lines)
    spread_arbs = find_spread_arbs(lines)
    all_arbs = ml_arbs + spread_arbs

    mgr = AlertManager(arb_min_margin=-5)
    alerts = mgr.check_arbitrage(all_arbs)

    if not all_arbs:
        print("No arbitrage opportunities found.")
        return 0

    for arb in all_arbs:
        tag = "*** ARB ***" if arb.profitable else "  near-arb "
        print(
            f"[{tag}] {arb.event}\n"
            f"  {arb.bet_type.value}: "
            f"{arb.side_a.sportsbook} vs {arb.side_b.sportsbook}\n"
            f"  Margin: {arb.margin:+.2f}%"
        )
        print()

    crit = [a for a in alerts if a.level.value == "critical"]
    if crit:
        print(f"=== {len(crit)} PROFITABLE ARB(S) DETECTED ===")
    return 0


def _cmd_moves(args) -> int:
    with LineStore(args.db) as store:
        all_lines = store.get_lines(
            event=args.event, limit=10000
        )

    if len(all_lines) < 2:
        print("Need at least 2 snapshots. Run 'fetch' again.")
        return 0

    # Compare only the two most recent snapshot timestamps.
    by_ts: dict = {}
    for ln in all_lines:
        by_ts.setdefault(ln.timestamp, []).append(ln)

    snapshots = sorted(by_ts)
    if len(snapshots) < 2:
        print("Need at least 2 snapshots. Run 'fetch' again.")
        return 0

    prev_ts, latest_ts = snapshots[-2], snapshots[-1]
    old_snap = by_ts[prev_ts]
    new_snap = by_ts[latest_ts]

    moves = detect_moves(old_snap, new_snap, threshold=args.threshold)

    if not moves:
        print("No line movements detected.")
        return 0

    mgr = AlertManager()
    alerts = mgr.check_movements(moves)

    for move in moves:
        print(
            f"{move.sportsbook} | {move.event} | "
            f"{move.bet_type.value}: "
            f"{move.old_line.home_value} -> "
            f"{move.new_line.home_value} "
            f"({move.change:+.1f}) [{move.direction}]"
        )

    if alerts:
        print(f"\n{len(alerts)} alert(s) triggered.")
    return 0


def _cmd_events(args) -> int:
    with LineStore(args.db) as store:
        events = store.get_events()
    if not events:
        print("No events tracked yet. Run 'fetch' first.")
        return 0
    for e in events:
        print(f"  {e}")
    return 0


def _cmd_slate(args) -> int:
    from line_tracker.models import BetType
    from line_tracker.services.slate_service import build_daily_slate_service

    sport_key = SPORTS[args.sport]
    with LineStore(args.db) as store:
        events = store.get_events_rich(sport=sport_key)
        lines_by_event: dict = {}
        for ev in events:
            eid = ev.get("api_event_id") or ev.get("event_id", "")
            all_lines: list = []
            for bt in BetType:
                try:
                    all_lines.extend(store.get_latest_for_api_event(eid, bt))
                except Exception:
                    pass
            if all_lines:
                lines_by_event[eid] = all_lines

        if not lines_by_event:
            print("No lines available. Run 'fetch' first.")
            return 0

        slate = build_daily_slate_service(
            lines_by_event,
            filters={"min_books": 4},
            mode=args.mode,
            store=store,
        )

    for tier_key in ("tier1a", "tier1b", "tier2", "tier3"):
        entries = slate.get(tier_key, [])
        if entries:
            print(f"\n--- {tier_key.upper()} ({len(entries)} picks) ---")
            for e in entries:
                print(
                    f"  {e.get('market', '?')} {e.get('selection', '?')}"
                    f"  @{e.get('best_sportsbook', '?')}"
                    f"  odds={e.get('best_odds', 0):+.0f}"
                    f"  edge_z={e.get('edge_z', 0):.2f}"
                )
    return 0


def _cmd_snapshot(args) -> int:
    from line_tracker.models import BetType
    from line_tracker.services.rec_snapshot_service import build_snapshot_rows
    from line_tracker.services.slate_service import build_daily_slate_service

    sport_key = SPORTS[args.sport]
    with LineStore(args.db) as store:
        events = store.get_events_rich(sport=sport_key)
        lines_by_event: dict = {}
        for ev in events:
            eid = ev.get("api_event_id") or ev.get("event_id", "")
            all_lines: list = []
            for bt in BetType:
                try:
                    all_lines.extend(store.get_latest_for_api_event(eid, bt))
                except Exception:
                    pass
            if all_lines:
                lines_by_event[eid] = all_lines

        if not lines_by_event:
            print("No lines available. Run 'fetch' first.")
            return 0

        slate = build_daily_slate_service(
            lines_by_event,
            filters={"min_books": 4},
            mode="Standard",
            store=store,
        )
        rows = build_snapshot_rows(slate, sport=sport_key)
        count = store.log_rec_snapshots(rows)

    print(f"Logged {count} snapshot rows.")
    return 0


def _cmd_cycle(args) -> int:
    from line_tracker.services.automation_service import run_cycle
    from line_tracker.services.ranking_service import format_picks_report

    sport_key = SPORTS[args.sport]
    with LineStore(args.db) as store:
        result = run_cycle(
            store,
            sport=sport_key,
            mode=args.mode,
            top_n=args.top_n,
            dry_run=args.dry_run,
        )

    print(f"Cycle completed at {result['cycle_ts']}")
    print(f"  Slate entries: {result['slate_entries']}")
    print(f"  Passed pruning: {result['pruned_count']}")
    print(f"  Snapshots logged: {result['snapshot_count']}")
    print(f"  Snapshots closed: {result['closed_count']}")
    print()
    print(format_picks_report(result["top_picks"]))
    return 0


def _cmd_import_outcomes(args) -> int:
    from line_tracker.services.outcome_service import (
        import_outcomes_csv,
        link_outcomes,
    )

    with LineStore(args.db) as store:
        imported = import_outcomes_csv(store, args.csv_file)
        linked = link_outcomes(store)

    print(f"Imported {imported} outcomes.")
    print(f"Linked {linked} snapshot rows to outcomes.")
    return 0


def _cmd_roi_report(args) -> int:
    from line_tracker.services.outcome_service import roi_report

    with LineStore(args.db) as store:
        report = roi_report(store)

    overall = report["overall"]
    print("=== Overall ROI ===")
    print(
        f"  Total: {overall['total']}  W: {overall['wins']}"
        f"  L: {overall['losses']}  P: {overall['pushes']}"
    )
    print(f"  Net units: {overall['net_units']:+.2f}")
    print(f"  ROI: {overall['roi_pct']:+.1f}%")

    if report["by_tier"]:
        print("\n=== ROI by Tier ===")
        for tier, stats in report["by_tier"].items():
            print(
                f"  {tier}: {stats['total']} picks"
                f"  W:{stats['wins']} L:{stats['losses']}"
                f"  ROI:{stats['roi_pct']:+.1f}%"
            )

    if report["by_alpha"]:
        print("\n=== ROI by Alpha ===")
        for alpha, stats in report["by_alpha"].items():
            print(
                f"  {alpha}: {stats['total']} picks"
                f"  W:{stats['wins']} L:{stats['losses']}"
                f"  ROI:{stats['roi_pct']:+.1f}%"
            )
    return 0


def _cmd_train_model(args) -> int:
    from line_tracker.services.clv_model_service import refresh_model

    with LineStore(args.db) as store:
        result = refresh_model(store)

    print(f"Model training: {result['status']}")
    if result.get("groups"):
        print(f"  Groups trained: {result['groups']}")
    if result.get("total_samples"):
        print(f"  Total samples: {result['total_samples']}")
    return 0


def _print_lines_table(lines) -> None:
    print(
        f"\n{'Book':<14} {'Event':<38} {'Type':<10} "
        f"{'Home':<8} {'Away':<8} {'H.Price':<8} {'A.Price':<8}"
    )
    print("-" * 104)
    for ln in lines:
        hp = f"{ln.home_price}" if ln.home_price is not None else ""
        ap = f"{ln.away_price}" if ln.away_price is not None else ""
        print(
            f"{ln.sportsbook:<14} "
            f"{ln.event[:36]:<38} "
            f"{ln.bet_type.value:<10} "
            f"{ln.home_value:<8} {ln.away_value:<8} "
            f"{hp:<8} {ap:<8}"
        )


if __name__ == "__main__":
    sys.exit(main())

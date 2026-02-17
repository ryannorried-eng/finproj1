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

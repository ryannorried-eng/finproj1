"""Arbitrage detection across sportsbooks."""

from __future__ import annotations

from dataclasses import dataclass

from line_tracker.models import BettingLine, BetType


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""

    event: str
    bet_type: BetType
    side_a: BettingLine  # e.g., home ML at book A
    side_b: BettingLine  # e.g., away ML at book B
    margin: float  # positive = guaranteed profit %

    @property
    def profitable(self) -> bool:
        return self.margin > 0


def find_moneyline_arbs(
    lines: list[BettingLine],
) -> list[ArbOpportunity]:
    """Find moneyline arb opportunities across sportsbooks.

    Compares the best home odds from one book against the best away
    odds from another. If implied probabilities sum to < 1, it's an arb.
    """
    ml_lines = [ln for ln in lines if ln.bet_type == BetType.MONEYLINE]

    events: dict[str, list[BettingLine]] = {}
    for ln in ml_lines:
        events.setdefault(ln.event, []).append(ln)

    arbs: list[ArbOpportunity] = []
    for event, event_lines in events.items():
        if len(event_lines) < 2:
            continue

        best_home = max(event_lines, key=lambda ln: ln.home_value)
        best_away = max(event_lines, key=lambda ln: ln.away_value)

        if best_home.sportsbook == best_away.sportsbook:
            continue

        home_prob = _implied_probability(best_home.home_value)
        away_prob = _implied_probability(best_away.away_value)
        total_prob = home_prob + away_prob

        margin = (1 - total_prob) * 100

        if margin > -5:  # show near-arbs too (within 5%)
            arbs.append(
                ArbOpportunity(
                    event=event,
                    bet_type=BetType.MONEYLINE,
                    side_a=best_home,
                    side_b=best_away,
                    margin=round(margin, 2),
                )
            )

    arbs.sort(key=lambda a: a.margin, reverse=True)
    return arbs


def find_spread_arbs(
    lines: list[BettingLine],
) -> list[ArbOpportunity]:
    """Find spread arb opportunities (different spread numbers)."""
    spread_lines = [ln for ln in lines if ln.bet_type == BetType.SPREAD]

    events: dict[str, list[BettingLine]] = {}
    for ln in spread_lines:
        events.setdefault(ln.event, []).append(ln)

    arbs: list[ArbOpportunity] = []
    for event, event_lines in events.items():
        if len(event_lines) < 2:
            continue

        # Best home spread = most positive (or least negative)
        best_home = max(event_lines, key=lambda ln: ln.home_value)
        # Best away spread = most positive
        best_away = max(event_lines, key=lambda ln: ln.away_value)

        if best_home.sportsbook == best_away.sportsbook:
            continue

        # Arb exists if spreads don't overlap
        # e.g., home +3.5 at book A, away +3.5 at book B (both positive)
        gap = best_home.home_value + best_away.away_value
        if gap > 0:
            arbs.append(
                ArbOpportunity(
                    event=event,
                    bet_type=BetType.SPREAD,
                    side_a=best_home,
                    side_b=best_away,
                    margin=round(gap, 1),
                )
            )

    arbs.sort(key=lambda a: a.margin, reverse=True)
    return arbs


def _implied_probability(american_odds: float) -> float:
    """Convert American odds to implied probability (0-1)."""
    if american_odds == 0:
        return 0.5  # EVEN
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 100 / (american_odds + 100)

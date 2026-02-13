"""Market-based 'Best Bet' recommendations using consensus (vig-free) probabilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median

from line_tracker.bet_slip import (
    american_to_decimal,
    breakeven_prob_from_american,
    ev_per_dollar,
    implied_prob_from_american,
)
from line_tracker.models import BettingLine, BetType


@dataclass
class BetRecommendation:
    """A single best-bet recommendation."""

    market: str  # "moneyline", "spread", "total"
    selection: str  # team name or "Over"/"Under"
    side: str  # "home", "away", "over", "under"
    line: float | None  # spread/total number, None for ML
    consensus_prob: float
    best_sportsbook: str
    best_odds: float  # American odds
    breakeven_prob: float  # implied prob at best_odds (breakeven threshold)
    ev: float  # expected value per $1 stake
    edge_pct: float  # (consensus_prob - breakeven_prob) * 100
    ev_per_100: float  # ev * 100 — dollar EV per $100 stake


def _remove_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove vig from a two-outcome market by normalizing implied probabilities.

    pA_no_vig = pA / (pA + pB)
    pB_no_vig = pB / (pA + pB)
    """
    pa = implied_prob_from_american(odds_a)
    pb = implied_prob_from_american(odds_b)
    total = pa + pb
    if total == 0:
        return 0.5, 0.5
    return pa / total, pb / total


def _compute_ev(consensus_prob: float, american_odds: float) -> float:
    """Compute expected value for a $1 stake.

    EV = p_consensus * (d - 1) - (1 - p_consensus) * 1
    where d is the decimal odds.
    """
    d = american_to_decimal(american_odds)
    return consensus_prob * (d - 1) - (1 - consensus_prob)


def _build_rec(
    *,
    market: str,
    selection: str,
    side: str,
    line: float | None,
    consensus_prob: float,
    best_sportsbook: str,
    best_odds: float,
) -> BetRecommendation:
    """Build a fully-populated BetRecommendation from core inputs."""
    be_prob = breakeven_prob_from_american(best_odds)
    ev = ev_per_dollar(consensus_prob, best_odds)
    edge = consensus_prob - be_prob
    return BetRecommendation(
        market=market,
        selection=selection,
        side=side,
        line=line,
        consensus_prob=round(consensus_prob, 4),
        best_sportsbook=best_sportsbook,
        best_odds=best_odds,
        breakeven_prob=round(be_prob, 4),
        ev=round(ev, 4),
        edge_pct=round(edge * 100, 2),
        ev_per_100=round(ev * 100, 2),
    )


def _mode_value(values: list[float]) -> float:
    """Return the most common value (mode).  Ties broken by Counter ordering."""
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _moneyline_recommendations(
    lines: list[BettingLine],
) -> list[BetRecommendation]:
    """Build moneyline best-bet recommendations for one event."""
    if len(lines) < 2:
        return []

    home_no_vig: list[float] = []
    away_no_vig: list[float] = []

    for ln in lines:
        ph, pa = _remove_vig(ln.home_value, ln.away_value)
        home_no_vig.append(ph)
        away_no_vig.append(pa)

    consensus_home = median(home_no_vig)
    consensus_away = median(away_no_vig)

    best_home_line = max(lines, key=lambda ln: ln.home_value)
    best_away_line = max(lines, key=lambda ln: ln.away_value)

    home_team = lines[0].home_team
    away_team = lines[0].away_team

    results: list[BetRecommendation] = []

    results.append(
        _build_rec(
            market="moneyline",
            selection=home_team,
            side="home",
            line=None,
            consensus_prob=consensus_home,
            best_sportsbook=best_home_line.sportsbook,
            best_odds=best_home_line.home_value,
        )
    )

    results.append(
        _build_rec(
            market="moneyline",
            selection=away_team,
            side="away",
            line=None,
            consensus_prob=consensus_away,
            best_sportsbook=best_away_line.sportsbook,
            best_odds=best_away_line.away_value,
        )
    )

    return results


def _spread_recommendations(
    lines: list[BettingLine],
) -> list[BetRecommendation]:
    """Build spread best-bet recommendations for one event.

    When spread lines differ between books (e.g., -3 at one book vs -3.5 at
    another), only the most common line (mode) is used for consensus probability
    calculation.  This ensures we compare apples-to-apples — probabilities are
    only meaningful at the same spread number.
    """
    priced = [
        ln for ln in lines if ln.home_price is not None and ln.away_price is not None
    ]
    if len(priced) < 2:
        return []

    mode_spread = _mode_value([ln.home_value for ln in priced])
    matching = [ln for ln in priced if ln.home_value == mode_spread]

    if len(matching) < 2:
        return []

    home_no_vig: list[float] = []
    away_no_vig: list[float] = []

    for ln in matching:
        ph, pa = _remove_vig(ln.home_price, ln.away_price)
        home_no_vig.append(ph)
        away_no_vig.append(pa)

    consensus_home = median(home_no_vig)
    consensus_away = median(away_no_vig)

    best_home = max(matching, key=lambda ln: ln.home_price)
    best_away = max(matching, key=lambda ln: ln.away_price)

    home_team = lines[0].home_team
    away_team = lines[0].away_team

    results: list[BetRecommendation] = []

    results.append(
        _build_rec(
            market="spread",
            selection=home_team,
            side="home",
            line=mode_spread,
            consensus_prob=consensus_home,
            best_sportsbook=best_home.sportsbook,
            best_odds=best_home.home_price,
        )
    )

    away_spread = matching[0].away_value
    results.append(
        _build_rec(
            market="spread",
            selection=away_team,
            side="away",
            line=away_spread,
            consensus_prob=consensus_away,
            best_sportsbook=best_away.sportsbook,
            best_odds=best_away.away_price,
        )
    )

    return results


def _total_recommendations(
    lines: list[BettingLine],
) -> list[BetRecommendation]:
    """Build total (over/under) best-bet recommendations for one event.

    When total lines differ between books (e.g., 45.5 at one book vs 46 at
    another), only the most common line (mode) is used for consensus probability
    calculation.
    """
    priced = [
        ln for ln in lines if ln.home_price is not None and ln.away_price is not None
    ]
    if len(priced) < 2:
        return []

    mode_total = _mode_value([ln.home_value for ln in priced])
    matching = [ln for ln in priced if ln.home_value == mode_total]

    if len(matching) < 2:
        return []

    over_no_vig: list[float] = []
    under_no_vig: list[float] = []

    for ln in matching:
        po, pu = _remove_vig(ln.home_price, ln.away_price)
        over_no_vig.append(po)
        under_no_vig.append(pu)

    consensus_over = median(over_no_vig)
    consensus_under = median(under_no_vig)

    best_over = max(matching, key=lambda ln: ln.home_price)
    best_under = max(matching, key=lambda ln: ln.away_price)

    results: list[BetRecommendation] = []

    results.append(
        _build_rec(
            market="total",
            selection="Over",
            side="over",
            line=mode_total,
            consensus_prob=consensus_over,
            best_sportsbook=best_over.sportsbook,
            best_odds=best_over.home_price,
        )
    )

    results.append(
        _build_rec(
            market="total",
            selection="Under",
            side="under",
            line=mode_total,
            consensus_prob=consensus_under,
            best_sportsbook=best_under.sportsbook,
            best_odds=best_under.away_price,
        )
    )

    return results


def recommend_best_bets(
    lines_for_event: list[BettingLine],
    top_n: int = 3,
) -> list[BetRecommendation]:
    """Return ranked best-bet recommendations (highest EV first).

    Analyzes moneyline, spread, and total markets for a single event,
    computes consensus (vig-free) probabilities via median across sportsbooks,
    and ranks candidate bets by expected value at the best available price.

    Parameters
    ----------
    lines_for_event:
        All betting lines for a single event across sportsbooks and markets.
    top_n:
        Maximum number of recommendations to return (default 3).

    Returns
    -------
    list[BetRecommendation]
        Recommendations sorted by EV descending, limited to *top_n*.
    """
    ml_lines = [ln for ln in lines_for_event if ln.bet_type == BetType.MONEYLINE]
    spread_lines = [ln for ln in lines_for_event if ln.bet_type == BetType.SPREAD]
    total_lines = [ln for ln in lines_for_event if ln.bet_type == BetType.TOTAL]

    recs: list[BetRecommendation] = []
    recs.extend(_moneyline_recommendations(ml_lines))
    recs.extend(_spread_recommendations(spread_lines))
    recs.extend(_total_recommendations(total_lines))

    recs.sort(key=lambda r: r.ev, reverse=True)
    return recs[:top_n]

"""Market-based 'Best Bet' recommendations using consensus (vig-free) probabilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from statistics import median, quantiles

from line_tracker.bet_slip import (
    american_to_decimal,
    breakeven_prob_from_american,
    ev_per_dollar,
    implied_prob_from_american,
)
from line_tracker.models import BettingLine, BetType

# ---------------------------------------------------------------------------
# Book weights — higher weight = more influence on consensus probability.
# Sharp / low-hold books get the highest weight; retail books get lower.
# ---------------------------------------------------------------------------
BOOK_WEIGHTS: dict[str, float] = {
    "Pinnacle": 3.0,
    "Circa": 2.5,
    "Bookmaker": 2.0,
    "BetOnline": 1.5,
    "BetMGM": 1.2,
    "DraftKings": 1.0,
    "FanDuel": 1.0,
    "Caesars": 1.0,
    "PointsBet": 0.8,
    "WynnBET": 0.8,
    "Unibet": 0.8,
    "SuperBook": 1.5,
    "BetRivers": 0.8,
}
_DEFAULT_WEIGHT = 1.0


def _weight_for_book(sportsbook: str) -> float:
    """Return the weight for a sportsbook, falling back to the default."""
    return BOOK_WEIGHTS.get(sportsbook, _DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# Recency weighting — fresher lines carry more influence.
# ---------------------------------------------------------------------------
RECENCY_HALF_LIFE_MIN = 60.0
_RECENCY_FLOOR = 0.01  # prevent underflow for very old timestamps


def _recency_multiplier(timestamp: datetime, now: datetime) -> float:
    """Exponential decay: exp(-age_minutes / RECENCY_HALF_LIFE_MIN).

    Returns a value in [_RECENCY_FLOOR, 1.0].  The floor prevents float
    underflow when lines are very old (e.g. in tests with fixed timestamps).
    """
    # Normalize tz-awareness so subtraction never fails.
    ts = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
    n = now.replace(tzinfo=None) if now.tzinfo else now
    age_minutes = max(0.0, (n - ts).total_seconds() / 60.0)
    return max(exp(-age_minutes / RECENCY_HALF_LIFE_MIN), _RECENCY_FLOOR)


def _strip_tz(dt: datetime) -> datetime:
    """Strip timezone info for safe naive subtraction."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _line_weight(sportsbook: str, timestamp: datetime, now: datetime) -> float:
    """Combined weight = book_weight * recency_multiplier."""
    return _weight_for_book(sportsbook) * _recency_multiplier(timestamp, now)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Compute a weighted median of *values* using *weights*.

    Algorithm: sort by value, walk through cumulative weight until we pass
    the 50 % mark of total weight, then return the corresponding value.
    If the midpoint falls exactly between two values we average them.
    """
    if len(values) == 1:
        return values[0]

    pairs = sorted(zip(values, weights), key=lambda vw: vw[0])
    total_w = sum(w for _, w in pairs)
    half = total_w / 2.0

    cumulative = 0.0
    for i, (val, w) in enumerate(pairs):
        cumulative += w
        if cumulative > half:
            return val
        if cumulative == half and i + 1 < len(pairs):
            return (val + pairs[i + 1][0]) / 2.0
    # Fallback (shouldn't happen with valid input)
    return pairs[-1][0]


@dataclass
class BetRecommendation:
    """A single best-bet recommendation."""

    market: str  # "moneyline", "spread", "total"
    selection: str  # team name or "Over"/"Under"
    side: str  # "home", "away", "over", "under"
    line: float | None  # spread/total number, None for ML
    consensus_prob: float  # weighted consensus (used for EV)
    best_sportsbook: str
    best_odds: float  # American odds
    breakeven_prob: float  # implied prob at best_odds (breakeven threshold)
    ev: float  # expected value per $1 stake
    edge_pct: float  # (consensus_prob - breakeven_prob) * 100
    ev_per_100: float  # ev * 100 — dollar EV per $100 stake
    confidence: str  # "High", "Medium", or "Low" — book agreement level
    unweighted_consensus_prob: float = 0.0  # plain median for debugging
    newest_update_age_min: float = 0.0  # minutes since most recent book update
    oldest_update_age_min: float = 0.0  # minutes since oldest book update
    books_used_count: int = 0  # books in the chosen line group
    total_books_count: int = 0  # total priced books for this market


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


def _confidence_label(probs: list[float]) -> str:
    """Classify sportsbook agreement as High / Medium / Low.

    Uses IQR (inter-quartile range, p75 − p25) of de-vigged probabilities
    across books.  Lower dispersion means books agree more closely on the
    true probability.

    Thresholds (on the 0–1 probability scale):
        IQR <= 0.03  → "High"   (<= 3 pp spread)
        IQR <= 0.06  → "Medium" (3–6 pp spread)
        IQR >  0.06  → "Low"    (> 6 pp spread)
    """
    if len(probs) < 2:
        return "Low"
    q1, _, q3 = quantiles(probs, n=4)
    iqr = q3 - q1
    if iqr <= 0.03:
        return "High"
    if iqr <= 0.06:
        return "Medium"
    return "Low"


def _build_rec(
    *,
    market: str,
    selection: str,
    side: str,
    line: float | None,
    consensus_prob: float,
    unweighted_consensus_prob: float,
    best_sportsbook: str,
    best_odds: float,
    side_probs: list[float],
    newest_update_age_min: float = 0.0,
    oldest_update_age_min: float = 0.0,
    books_used_count: int = 0,
    total_books_count: int = 0,
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
        confidence=_confidence_label(side_probs),
        unweighted_consensus_prob=round(unweighted_consensus_prob, 4),
        newest_update_age_min=round(newest_update_age_min, 1),
        oldest_update_age_min=round(oldest_update_age_min, 1),
        books_used_count=books_used_count,
        total_books_count=total_books_count,
    )


def _mode_value(values: list[float]) -> float:
    """Return the most common value (mode).  Ties broken by Counter ordering."""
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _best_line_group(values: list[float]) -> float:
    """Choose the best line value for consensus.

    Selection criteria (in order):
      1. Most books offering that line value.
      2. Tie-break: closest to the median of *all* line values.

    Returns the chosen line value.
    """
    counter = Counter(values)
    max_count = counter.most_common(1)[0][1]
    # All line values that are tied for the highest count
    tied = [val for val, cnt in counter.items() if cnt == max_count]
    if len(tied) == 1:
        return tied[0]
    # Break tie: pick the value closest to the overall median
    med = median(values)
    return min(tied, key=lambda v: abs(v - med))


def _moneyline_recommendations(
    lines: list[BettingLine],
    now: datetime,
) -> list[BetRecommendation]:
    """Build moneyline best-bet recommendations for one event."""
    if len(lines) < 2:
        return []

    home_no_vig: list[float] = []
    away_no_vig: list[float] = []
    weights: list[float] = []

    for ln in lines:
        ph, pa = _remove_vig(ln.home_value, ln.away_value)
        home_no_vig.append(ph)
        away_no_vig.append(pa)
        weights.append(_line_weight(ln.sportsbook, ln.timestamp, now))

    # Unweighted (plain median) — kept for debugging
    unweighted_home = median(home_no_vig)
    unweighted_away = median(away_no_vig)

    # Weighted median — used for EV calculation
    consensus_home = _weighted_median(home_no_vig, weights)
    consensus_away = _weighted_median(away_no_vig, weights)

    best_home_line = max(lines, key=lambda ln: ln.home_value)
    best_away_line = max(lines, key=lambda ln: ln.away_value)

    home_team = lines[0].home_team
    away_team = lines[0].away_team

    ages = [(_strip_tz(now) - _strip_tz(ln.timestamp)).total_seconds() / 60.0 for ln in lines]
    newest_age = max(0.0, min(ages))
    oldest_age = max(0.0, max(ages))

    results: list[BetRecommendation] = []

    n_books = len(lines)

    results.append(
        _build_rec(
            market="moneyline",
            selection=home_team,
            side="home",
            line=None,
            consensus_prob=consensus_home,
            unweighted_consensus_prob=unweighted_home,
            best_sportsbook=best_home_line.sportsbook,
            best_odds=best_home_line.home_value,
            side_probs=home_no_vig,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=n_books,
            total_books_count=n_books,
        )
    )

    results.append(
        _build_rec(
            market="moneyline",
            selection=away_team,
            side="away",
            line=None,
            consensus_prob=consensus_away,
            unweighted_consensus_prob=unweighted_away,
            best_sportsbook=best_away_line.sportsbook,
            best_odds=best_away_line.away_value,
            side_probs=away_no_vig,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=n_books,
            total_books_count=n_books,
        )
    )

    return results


def _spread_recommendations(
    lines: list[BettingLine],
    now: datetime,
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

    chosen_spread = _best_line_group([ln.home_value for ln in priced])
    matching = [ln for ln in priced if ln.home_value == chosen_spread]

    if len(matching) < 2:
        return []

    total_books = len(priced)
    books_used = len(matching)

    home_no_vig: list[float] = []
    away_no_vig: list[float] = []
    weights: list[float] = []

    for ln in matching:
        ph, pa = _remove_vig(ln.home_price, ln.away_price)
        home_no_vig.append(ph)
        away_no_vig.append(pa)
        weights.append(_line_weight(ln.sportsbook, ln.timestamp, now))

    unweighted_home = median(home_no_vig)
    unweighted_away = median(away_no_vig)

    consensus_home = _weighted_median(home_no_vig, weights)
    consensus_away = _weighted_median(away_no_vig, weights)

    best_home = max(matching, key=lambda ln: ln.home_price)
    best_away = max(matching, key=lambda ln: ln.away_price)

    home_team = lines[0].home_team
    away_team = lines[0].away_team

    ages = [(_strip_tz(now) - _strip_tz(ln.timestamp)).total_seconds() / 60.0 for ln in matching]
    newest_age = max(0.0, min(ages))
    oldest_age = max(0.0, max(ages))

    results: list[BetRecommendation] = []

    results.append(
        _build_rec(
            market="spread",
            selection=home_team,
            side="home",
            line=chosen_spread,
            consensus_prob=consensus_home,
            unweighted_consensus_prob=unweighted_home,
            best_sportsbook=best_home.sportsbook,
            best_odds=best_home.home_price,
            side_probs=home_no_vig,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
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
            unweighted_consensus_prob=unweighted_away,
            best_sportsbook=best_away.sportsbook,
            best_odds=best_away.away_price,
            side_probs=away_no_vig,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
        )
    )

    return results


def _total_recommendations(
    lines: list[BettingLine],
    now: datetime,
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

    chosen_total = _best_line_group([ln.home_value for ln in priced])
    matching = [ln for ln in priced if ln.home_value == chosen_total]

    if len(matching) < 2:
        return []

    total_books = len(priced)
    books_used = len(matching)

    over_no_vig: list[float] = []
    under_no_vig: list[float] = []
    weights: list[float] = []

    for ln in matching:
        po, pu = _remove_vig(ln.home_price, ln.away_price)
        over_no_vig.append(po)
        under_no_vig.append(pu)
        weights.append(_line_weight(ln.sportsbook, ln.timestamp, now))

    unweighted_over = median(over_no_vig)
    unweighted_under = median(under_no_vig)

    consensus_over = _weighted_median(over_no_vig, weights)
    consensus_under = _weighted_median(under_no_vig, weights)

    best_over = max(matching, key=lambda ln: ln.home_price)
    best_under = max(matching, key=lambda ln: ln.away_price)

    ages = [(_strip_tz(now) - _strip_tz(ln.timestamp)).total_seconds() / 60.0 for ln in matching]
    newest_age = max(0.0, min(ages))
    oldest_age = max(0.0, max(ages))

    results: list[BetRecommendation] = []

    results.append(
        _build_rec(
            market="total",
            selection="Over",
            side="over",
            line=chosen_total,
            consensus_prob=consensus_over,
            unweighted_consensus_prob=unweighted_over,
            best_sportsbook=best_over.sportsbook,
            best_odds=best_over.home_price,
            side_probs=over_no_vig,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
        )
    )

    results.append(
        _build_rec(
            market="total",
            selection="Under",
            side="under",
            line=chosen_total,
            consensus_prob=consensus_under,
            unweighted_consensus_prob=unweighted_under,
            best_sportsbook=best_under.sportsbook,
            best_odds=best_under.away_price,
            side_probs=under_no_vig,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
        )
    )

    return results


def recommend_best_bets(
    lines_for_event: list[BettingLine],
    top_n: int = 3,
    now: datetime | None = None,
) -> list[BetRecommendation]:
    """Return ranked best-bet recommendations (highest EV first).

    Analyzes moneyline, spread, and total markets for a single event,
    computes consensus (vig-free) probabilities via weighted median across
    sportsbooks (sharper books weigh more, fresher lines weigh more), and
    ranks candidate bets by expected value at the best available price.

    Parameters
    ----------
    lines_for_event:
        All betting lines for a single event across sportsbooks and markets.
    top_n:
        Maximum number of recommendations to return (default 3).
    now:
        Reference time for recency weighting.  Defaults to ``datetime.now(timezone.utc)``.

    Returns
    -------
    list[BetRecommendation]
        Recommendations sorted by EV descending, limited to *top_n*.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ml_lines = [ln for ln in lines_for_event if ln.bet_type == BetType.MONEYLINE]
    spread_lines = [ln for ln in lines_for_event if ln.bet_type == BetType.SPREAD]
    total_lines = [ln for ln in lines_for_event if ln.bet_type == BetType.TOTAL]

    recs: list[BetRecommendation] = []
    recs.extend(_moneyline_recommendations(ml_lines, now))
    recs.extend(_spread_recommendations(spread_lines, now))
    recs.extend(_total_recommendations(total_lines, now))

    recs.sort(key=lambda r: r.ev, reverse=True)
    return recs[:top_n]

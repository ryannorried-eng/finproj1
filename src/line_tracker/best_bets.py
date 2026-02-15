"""Market-based 'Best Bet' recommendations using consensus (vig-free) probabilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp
from statistics import median, quantiles, stdev

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

# Tier downgrade map when a market is flagged as unstable.
_TIER_DOWNGRADE = {"Elite": "Strong", "Strong": "Moderate", "Moderate": "Thin"}


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


# ---------------------------------------------------------------------------
# Weight capping — prevent any single book from dominating consensus.
# ---------------------------------------------------------------------------
_MAX_BOOK_WEIGHT_SHARE = 0.40  # no book may exceed 40% of total weight


def _cap_weights(weights: list[float]) -> list[float]:
    """Normalize *weights* to sum to 1, cap each at _MAX_BOOK_WEIGHT_SHARE.

    Uses iterative redistribution: freeze any weight that exceeds the cap,
    then redistribute the remaining budget proportionally among unfrozen
    weights.  Repeats until no weight exceeds the cap.

    The cap is only applied when there are 3+ books — with fewer books,
    equal distribution already exceeds the cap (e.g., 50% each with 2 books),
    so capping would be counterproductive.

    This prevents a single very-sharp or very-fresh book from contributing
    more than 40 % of the total influence on consensus probability.
    """
    n = len(weights)
    if n == 0:
        return []
    total = sum(weights)
    if total == 0:
        return [1.0 / n] * n

    normed = [w / total for w in weights]

    # Cap is only meaningful when n >= ceil(1/cap).  With cap=0.40 that
    # means n >= 3.  With fewer books, just return normalized weights.
    if n < 3:
        return normed

    cap = _MAX_BOOK_WEIGHT_SHARE
    frozen = [False] * n

    # Iterate until no unfrozen weight exceeds the cap
    for _ in range(n):
        changed = False
        for i in range(n):
            if frozen[i]:
                continue
            if normed[i] > cap:
                normed[i] = cap
                frozen[i] = True
                changed = True

        if not changed:
            break

        # Redistribute: unfrozen weights share the remaining budget
        frozen_sum = sum(normed[i] for i in range(n) if frozen[i])
        remaining = 1.0 - frozen_sum
        unfrozen_raw = sum(weights[i] for i in range(n) if not frozen[i])
        if unfrozen_raw > 0:
            for i in range(n):
                if not frozen[i]:
                    normed[i] = (weights[i] / unfrozen_raw) * remaining

    return normed


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
    quality_score: int = 0  # composite 0–100 bet quality score
    quality_tier: str = ""  # "Elite", "Strong", "Moderate", or "Thin"
    edge_score: float = 0.0  # 0–100 subscore for edge size
    agreement_score: float = 0.0  # 0–100 subscore for book agreement
    coverage_score: float = 0.0  # 0–100 subscore for market coverage
    freshness_score: float = 0.0  # 0–100 subscore for data freshness
    outlier_filtered: bool = False  # True if outlier books were removed
    market_unstable: bool = False  # True if too many books were filtered out
    skipped_reason: str = ""  # non-empty if this rec was skipped (unsupported)
    book_holds: dict[str, float] = field(default_factory=dict)  # implied hold% per book
    market_hold_median: float = 0.0  # median hold% across books
    market_volatility_sigma: float = 0.0  # std dev of devigged probs
    robust_sigma: float = 0.0  # IQR / 1.349
    edge_z: float = 0.0  # edge / max(robust_sigma, 0.01)


# ---------------------------------------------------------------------------
# Three-way market guard — sports whose moneylines have a draw outcome.
# ---------------------------------------------------------------------------
_THREE_WAY_SPORT_PREFIXES = ("soccer",)


def _is_three_way_market(lines: list[BettingLine]) -> bool:
    """Return True if *lines* belong to a 3-outcome moneyline market.

    Detection: the sport key starts with a known 3-way prefix (e.g.
    ``"soccer_epl"``) **and** the bet type is ``MONEYLINE``.  Spread and
    total markets remain 2-outcome even in soccer, so they are not
    affected.
    """
    if not lines:
        return False
    first = lines[0]
    if first.bet_type != BetType.MONEYLINE:
        return False
    sport = first.sport.lower()
    return any(sport.startswith(prefix) for prefix in _THREE_WAY_SPORT_PREFIXES)


# ---------------------------------------------------------------------------
# Outlier filtering — remove books whose de-vigged probability deviates
# too far from the median before computing consensus.
# ---------------------------------------------------------------------------
_OUTLIER_REL_THRESHOLD = 0.15  # relative deviation threshold
_OUTLIER_ABS_FLOOR = 0.01  # minimum allowable probability
_OUTLIER_ABS_CEIL = 0.99  # maximum allowable probability
_OUTLIER_DROP_RATIO = 0.30  # if > 30% of books filtered → market_unstable
_OUTLIER_MIN_BOOKS = 4  # if books_after < 4 → market_unstable


def _filter_outliers(
    probs: list[float],
) -> tuple[list[int], bool]:
    """Identify outlier indices in a list of de-vigged probabilities.

    Returns a tuple of (keep_indices, market_unstable).
    keep_indices: indices of probabilities that passed filtering.
    market_unstable: True if too many books were removed or too few remain.
    """
    n = len(probs)
    if n < 2:
        return list(range(n)), False

    p_med = median(probs)

    keep: list[int] = []
    for i, p in enumerate(probs):
        # Filter if outside absolute bounds
        if p < _OUTLIER_ABS_FLOOR or p > _OUTLIER_ABS_CEIL:
            continue
        # Filter if relative deviation from median is too large
        rel_dev = abs(p - p_med) / max(p_med, 1e-6)
        if rel_dev > _OUTLIER_REL_THRESHOLD:
            continue
        keep.append(i)

    # If filtering removed everything, fall back to all indices
    if not keep:
        return list(range(n)), True

    removed_count = n - len(keep)
    removed_ratio = removed_count / n
    unstable = removed_ratio > _OUTLIER_DROP_RATIO or len(keep) < _OUTLIER_MIN_BOOKS

    return keep, unstable


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


# ---------------------------------------------------------------------------
# Bet Quality Score — composite 0–100 score with subscores
# ---------------------------------------------------------------------------

_EDGE_BREAKPOINTS: list[tuple[float, float]] = [
    (0.0, 0),
    (0.5, 35),
    (1.0, 55),
    (2.0, 75),
    (3.0, 85),
    (5.0, 95),
    (7.0, 100),
]


def _edge_score(edge_pct: float) -> float:
    """Map *edge_pct* (percentage points) to a 0–100 score.

    Uses piecewise-linear interpolation through ``_EDGE_BREAKPOINTS``.
    Values beyond 7 % are capped at 100.
    """
    if edge_pct <= 0:
        return 0.0
    for i in range(len(_EDGE_BREAKPOINTS) - 1):
        x0, y0 = _EDGE_BREAKPOINTS[i]
        x1, y1 = _EDGE_BREAKPOINTS[i + 1]
        if edge_pct <= x1:
            t = (edge_pct - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 100.0


def _agreement_score(
    side_probs: list[float],
    books_used: int,
    stalest_age_min: float,
) -> float:
    """Score sportsbook agreement on a 0–100 scale.

    Combines IQR dispersion (inter-quartile range) with standard-deviation
    penalties so that distributions with heavy tails score lower even when
    IQR looks tight.

    Penalties:
      - std > 0.08  →  −25
      - std > 0.05  →  −15
      - books_used < 6  →  −10
      - stalest_age_min > 60  →  −10
    """
    # Base from IQR
    if len(side_probs) < 2:
        base = 40.0
    else:
        q1, _, q3 = quantiles(side_probs, n=4)
        iqr = q3 - q1
        if iqr <= 0.03:
            base = 90.0
        elif iqr <= 0.06:
            base = 70.0
        else:
            base = 40.0

    # Std-deviation penalty (captures tail spread IQR may miss)
    if len(side_probs) >= 2:
        sd = stdev(side_probs)
        if sd > 0.08:
            base -= 25.0
        elif sd > 0.05:
            base -= 15.0

    # Penalties
    if books_used < 6:
        base -= 10.0
    if stalest_age_min > 60:
        base -= 10.0

    return max(0.0, min(base, 100.0))


_THIN_MARKET_BOOKS = 5  # markets with fewer total books are "thin"
_THIN_MARKET_COVERAGE_CAP = 70.0  # max coverage_score for thin markets


def _coverage_score(books_used: int, books_total: int) -> float:
    """Score market coverage (books_used / books_total) on 0–100.

    When *books_total* < 5 the market is considered thin and the score
    is capped at 70 — even perfect used/total ratio shouldn't claim
    full coverage when the sample is small.
    """
    raw = max(0.0, min(books_used / max(books_total, 1) * 100.0, 100.0))
    if books_total < _THIN_MARKET_BOOKS:
        raw = min(raw, _THIN_MARKET_COVERAGE_CAP)
    return raw


def _freshness_score(freshest_age_min: float, stalest_age_min: float) -> float:
    """Score data freshness on 0–100 based on age of lines."""
    # Primary driver is the stalest line
    if stalest_age_min <= 10:
        return 95.0
    if stalest_age_min <= 30:
        return 75.0
    if stalest_age_min <= 60:
        return 60.0
    return 40.0


_QW_EDGE = 0.45
_QW_AGREEMENT = 0.25
_QW_COVERAGE = 0.20
_QW_FRESHNESS = 0.10


def _quality_score(
    edge: float,
    agreement: float,
    coverage: float,
    freshness: float,
) -> int:
    """Combine subscores into a single 0–100 integer quality score."""
    raw = _QW_EDGE * edge + _QW_AGREEMENT * agreement + _QW_COVERAGE * coverage + _QW_FRESHNESS * freshness
    return max(0, min(round(raw), 100))


def _quality_tier(score: int) -> str:
    """Map a 0–100 quality score to a human-readable tier label."""
    if score >= 85:
        return "Elite"
    if score >= 70:
        return "Strong"
    if score >= 55:
        return "Moderate"
    return "Thin"


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
    outlier_filtered: bool = False,
    market_unstable: bool = False,
    book_holds: dict[str, float] | None = None,
) -> BetRecommendation:
    """Build a fully-populated BetRecommendation from core inputs."""
    be_prob = breakeven_prob_from_american(best_odds)
    ev = ev_per_dollar(consensus_prob, best_odds)
    edge = consensus_prob - be_prob
    edge_pct_val = round(edge * 100, 2)

    # Quality subscores
    e_score = round(_edge_score(edge_pct_val), 1)
    a_score = round(
        _agreement_score(side_probs, books_used_count, oldest_update_age_min), 1
    )
    c_score = round(_coverage_score(books_used_count, total_books_count), 1)
    f_score = round(
        _freshness_score(newest_update_age_min, oldest_update_age_min), 1
    )

    # If market is unstable, cap agreement_score and downgrade tier
    if market_unstable:
        a_score = min(a_score, 60.0)

    q_score = _quality_score(e_score, a_score, c_score, f_score)
    q_tier = _quality_tier(q_score)

    # Downgrade tier by one level when market is unstable
    if market_unstable:
        q_tier = _TIER_DOWNGRADE.get(q_tier, q_tier)

    # Thin book usage: fewer than 4 books used → cap score and tier
    if books_used_count < 4:
        q_score = min(q_score, 55)
        if q_tier in ("Elite", "Strong"):
            q_tier = "Moderate"

    # --- New consensus metrics ---
    holds = book_holds or {}
    hold_values = list(holds.values())
    mkt_hold_med = round(median(hold_values), 2) if hold_values else 0.0

    # Volatility of devigged probabilities (filtered set)
    if len(side_probs) >= 2:
        vol_sigma = round(stdev(side_probs), 4)
        q1, _, q3 = quantiles(side_probs, n=4)
        r_sigma = round((q3 - q1) / 1.349, 4)
    else:
        vol_sigma = 0.0
        r_sigma = 0.0

    ez = round(edge / max(r_sigma, 0.01), 2)

    # Confidence derived from edge_z thresholds
    if ez >= 2.5:
        confidence = "High"
    elif ez >= 1.5:
        confidence = "Medium"
    else:
        confidence = "Low"

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
        edge_pct=edge_pct_val,
        ev_per_100=round(ev * 100, 2),
        confidence=confidence,
        unweighted_consensus_prob=round(unweighted_consensus_prob, 4),
        newest_update_age_min=round(newest_update_age_min, 1),
        oldest_update_age_min=round(oldest_update_age_min, 1),
        books_used_count=books_used_count,
        total_books_count=total_books_count,
        quality_score=q_score,
        quality_tier=q_tier,
        edge_score=e_score,
        agreement_score=a_score,
        coverage_score=c_score,
        freshness_score=f_score,
        outlier_filtered=outlier_filtered,
        market_unstable=market_unstable,
        book_holds=holds,
        market_hold_median=mkt_hold_med,
        market_volatility_sigma=vol_sigma,
        robust_sigma=r_sigma,
        edge_z=ez,
    )


def _mode_value(values: list[float]) -> float:
    """Return the most common value (mode).  Ties broken by Counter ordering."""
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _best_line_group(
    values: list[float],
    weights: list[float] | None = None,
) -> float:
    """Choose the best line value for consensus.

    Selection criteria (in order):
      1. Most books offering that line value.
      2. If the top two groups are within 1 book of each other (near-tie),
         pick the line closest to the weighted median of *all* line values
         (or plain median when *weights* is ``None``).

    Returns the chosen line value.
    """
    counter = Counter(values)
    ranked = counter.most_common()         # [(value, count), ...] desc by count
    top_count = ranked[0][1]

    # Near-tie: top-2 groups differ by at most 1 book
    if len(ranked) >= 2 and top_count - ranked[1][1] <= 1:
        near_tie_threshold = top_count - 1
        candidates = [val for val, cnt in ranked if cnt >= near_tie_threshold]
    else:
        candidates = [ranked[0][0]]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break: closest to the (weighted) median of all line values
    if weights is not None and len(weights) == len(values):
        med = _weighted_median(values, weights)
    else:
        med = median(values)
    return min(candidates, key=lambda v: abs(v - med))


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

    # Unweighted (plain median) — kept for debugging (computed before filtering)
    unweighted_home = median(home_no_vig)
    unweighted_away = median(away_no_vig)

    # Outlier filtering — use home side to determine which books to keep
    keep_idx, unstable_home = _filter_outliers(home_no_vig)
    _, unstable_away = _filter_outliers(away_no_vig)
    market_unstable = unstable_home or unstable_away
    outlier_filtered = len(keep_idx) < len(home_no_vig)

    # Apply filter to both sides consistently
    f_home = [home_no_vig[i] for i in keep_idx]
    f_away = [away_no_vig[i] for i in keep_idx]
    f_weights = _cap_weights([weights[i] for i in keep_idx])

    # Weighted median — used for EV calculation (on filtered set)
    consensus_home = _weighted_median(f_home, f_weights)
    consensus_away = _weighted_median(f_away, f_weights)

    best_home_line = max(lines, key=lambda ln: ln.home_value)
    best_away_line = max(lines, key=lambda ln: ln.away_value)

    home_team = lines[0].home_team
    away_team = lines[0].away_team

    f_lines = [lines[i] for i in keep_idx]
    ages = [(_strip_tz(now) - _strip_tz(ln.timestamp)).total_seconds() / 60.0 for ln in f_lines]
    newest_age = max(0.0, min(ages))
    oldest_age = max(0.0, max(ages))

    # Implied hold% per sportsbook (all books, pre-filter)
    book_holds: dict[str, float] = {}
    for ln in lines:
        pa_raw = implied_prob_from_american(ln.home_value)
        pb_raw = implied_prob_from_american(ln.away_value)
        book_holds[ln.sportsbook] = round((pa_raw + pb_raw - 1) * 100, 2)

    results: list[BetRecommendation] = []

    n_total = len(lines)
    n_used = len(keep_idx)

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
            side_probs=f_home,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=n_used,
            total_books_count=n_total,
            outlier_filtered=outlier_filtered,
            market_unstable=market_unstable,
            book_holds=book_holds,
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
            side_probs=f_away,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=n_used,
            total_books_count=n_total,
            outlier_filtered=outlier_filtered,
            market_unstable=market_unstable,
            book_holds=book_holds,
        )
    )

    return results


def _spread_recommendations(
    lines: list[BettingLine],
    now: datetime,
) -> list[BetRecommendation]:
    """Build spread best-bet recommendations for one event.

    When spread lines differ between books (e.g., -3 at one book vs -3.5 at
    another), only the most common line (or near-tie best) is used for consensus
    probability calculation.  This ensures we compare apples-to-apples —
    probabilities are only meaningful at the same spread number.
    """
    priced = [
        ln for ln in lines if ln.home_price is not None and ln.away_price is not None
    ]
    if len(priced) < 2:
        return []

    line_values = [ln.home_value for ln in priced]
    line_weights = [_line_weight(ln.sportsbook, ln.timestamp, now) for ln in priced]
    chosen_spread = _best_line_group(line_values, line_weights)
    matching = [ln for ln in priced if ln.home_value == chosen_spread]

    if len(matching) < 2:
        return []

    total_books = len(priced)

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

    # Outlier filtering
    keep_idx, unstable_home = _filter_outliers(home_no_vig)
    _, unstable_away = _filter_outliers(away_no_vig)
    market_unstable = unstable_home or unstable_away
    outlier_filtered = len(keep_idx) < len(home_no_vig)

    f_home = [home_no_vig[i] for i in keep_idx]
    f_away = [away_no_vig[i] for i in keep_idx]
    f_weights = _cap_weights([weights[i] for i in keep_idx])
    f_matching = [matching[i] for i in keep_idx]

    books_used = len(keep_idx)

    consensus_home = _weighted_median(f_home, f_weights)
    consensus_away = _weighted_median(f_away, f_weights)

    best_home = max(f_matching, key=lambda ln: ln.home_price)
    best_away = max(f_matching, key=lambda ln: ln.away_price)

    home_team = lines[0].home_team
    away_team = lines[0].away_team

    ages = [(_strip_tz(now) - _strip_tz(ln.timestamp)).total_seconds() / 60.0 for ln in f_matching]
    newest_age = max(0.0, min(ages))
    oldest_age = max(0.0, max(ages))

    # Implied hold% per sportsbook (matching lines, pre-filter)
    book_holds: dict[str, float] = {}
    for ln in matching:
        pa_raw = implied_prob_from_american(ln.home_price)
        pb_raw = implied_prob_from_american(ln.away_price)
        book_holds[ln.sportsbook] = round((pa_raw + pb_raw - 1) * 100, 2)

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
            side_probs=f_home,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
            outlier_filtered=outlier_filtered,
            market_unstable=market_unstable,
            book_holds=book_holds,
        )
    )

    away_spread = f_matching[0].away_value
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
            side_probs=f_away,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
            outlier_filtered=outlier_filtered,
            market_unstable=market_unstable,
            book_holds=book_holds,
        )
    )

    return results


def _total_recommendations(
    lines: list[BettingLine],
    now: datetime,
) -> list[BetRecommendation]:
    """Build total (over/under) best-bet recommendations for one event.

    When total lines differ between books (e.g., 45.5 at one book vs 46 at
    another), only the most common line (or near-tie best) is used for consensus
    probability calculation.
    """
    priced = [
        ln for ln in lines if ln.home_price is not None and ln.away_price is not None
    ]
    if len(priced) < 2:
        return []

    line_values = [ln.home_value for ln in priced]
    line_weights = [_line_weight(ln.sportsbook, ln.timestamp, now) for ln in priced]
    chosen_total = _best_line_group(line_values, line_weights)
    matching = [ln for ln in priced if ln.home_value == chosen_total]

    if len(matching) < 2:
        return []

    total_books = len(priced)

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

    # Outlier filtering
    keep_idx, unstable_over = _filter_outliers(over_no_vig)
    _, unstable_under = _filter_outliers(under_no_vig)
    market_unstable = unstable_over or unstable_under
    outlier_filtered = len(keep_idx) < len(over_no_vig)

    f_over = [over_no_vig[i] for i in keep_idx]
    f_under = [under_no_vig[i] for i in keep_idx]
    f_weights = _cap_weights([weights[i] for i in keep_idx])
    f_matching = [matching[i] for i in keep_idx]

    books_used = len(keep_idx)

    consensus_over = _weighted_median(f_over, f_weights)
    consensus_under = _weighted_median(f_under, f_weights)

    best_over = max(f_matching, key=lambda ln: ln.home_price)
    best_under = max(f_matching, key=lambda ln: ln.away_price)

    ages = [(_strip_tz(now) - _strip_tz(ln.timestamp)).total_seconds() / 60.0 for ln in f_matching]
    newest_age = max(0.0, min(ages))
    oldest_age = max(0.0, max(ages))

    # Implied hold% per sportsbook (matching lines, pre-filter)
    book_holds: dict[str, float] = {}
    for ln in matching:
        pa_raw = implied_prob_from_american(ln.home_price)
        pb_raw = implied_prob_from_american(ln.away_price)
        book_holds[ln.sportsbook] = round((pa_raw + pb_raw - 1) * 100, 2)

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
            side_probs=f_over,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
            outlier_filtered=outlier_filtered,
            market_unstable=market_unstable,
            book_holds=book_holds,
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
            side_probs=f_under,
            newest_update_age_min=newest_age,
            oldest_update_age_min=oldest_age,
            books_used_count=books_used,
            total_books_count=total_books,
            outlier_filtered=outlier_filtered,
            market_unstable=market_unstable,
            book_holds=book_holds,
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

    # Guard: skip 3-way moneylines (e.g. soccer with draw outcome).
    # Two-way normalization is invalid when a third outcome exists.
    if _is_three_way_market(ml_lines):
        recs.append(
            BetRecommendation(
                market="moneyline",
                selection="",
                side="",
                line=None,
                consensus_prob=0.0,
                best_sportsbook="",
                best_odds=0.0,
                breakeven_prob=0.0,
                ev=0.0,
                edge_pct=0.0,
                ev_per_100=0.0,
                confidence="Low",
                skipped_reason="3-way market (draw outcome); 2-way model unsupported",
            )
        )
    else:
        recs.extend(_moneyline_recommendations(ml_lines, now))

    recs.extend(_spread_recommendations(spread_lines, now))
    recs.extend(_total_recommendations(total_lines, now))

    recs.sort(key=lambda r: r.ev, reverse=True)
    return recs[:top_n]

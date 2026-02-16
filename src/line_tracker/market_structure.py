"""Market structure analytics for evaluating market efficiency."""

from __future__ import annotations

from collections import Counter
from statistics import median, quantiles, stdev

from line_tracker.best_bets import (
    BOOK_WEIGHTS,
    _remove_vig,
    _weighted_median,
)
from line_tracker.models import BettingLine, BetType

# Books with weight >= 1.5 are considered sharp.
_SHARP_THRESHOLD = 1.5
_DEFAULT_WEIGHT = 1.0

SHARP_BOOKS: set[str] = {
    book for book, w in BOOK_WEIGHTS.items() if w >= _SHARP_THRESHOLD
}


def _is_sharp(sportsbook: str) -> bool:
    return sportsbook in SHARP_BOOKS


def _book_weight(sportsbook: str) -> float:
    return BOOK_WEIGHTS.get(sportsbook, _DEFAULT_WEIGHT)


# ------------------------------------------------------------------
# Line dispersion (spread / total markets)
# ------------------------------------------------------------------

def line_dispersion(lines: list[BettingLine]) -> dict[float, int]:
    """Count how many books offer each line value.

    For spread markets uses home_value; for totals uses home_value.
    Returns {line_value: count} sorted by value.
    """
    if not lines:
        return {}
    vals = [ln.home_value for ln in lines]
    counts = Counter(vals)
    return dict(sorted(counts.items()))


# ------------------------------------------------------------------
# Sharp vs retail divergence
# ------------------------------------------------------------------

def sharp_retail_divergence(
    lines: list[BettingLine],
) -> dict:
    """Compute consensus probability for sharp books vs retail books.

    Returns a dict with:
        sharp_consensus, retail_consensus, divergence,
        sharp_books (list), retail_books (list).
    Returns None values when either group has < 2 books.
    """
    bt = lines[0].bet_type if lines else None

    sharp_lines = [ln for ln in lines if _is_sharp(ln.sportsbook)]
    retail_lines = [ln for ln in lines if not _is_sharp(ln.sportsbook)]

    result: dict = {
        "sharp_books": [ln.sportsbook for ln in sharp_lines],
        "retail_books": [ln.sportsbook for ln in retail_lines],
        "sharp_consensus": None,
        "retail_consensus": None,
        "divergence": None,
    }

    if len(sharp_lines) < 2 or len(retail_lines) < 2:
        return result

    def _consensus(subset: list[BettingLine]) -> float:
        """Weighted-median consensus for home/over side."""
        probs = []
        weights = []
        for ln in subset:
            if bt == BetType.MONEYLINE:
                p, _ = _remove_vig(ln.home_value, ln.away_value)
            else:
                if ln.home_price is None or ln.away_price is None:
                    continue
                p, _ = _remove_vig(ln.home_price, ln.away_price)
            probs.append(p)
            weights.append(_book_weight(ln.sportsbook))
        if not probs:
            return 0.0
        total_w = sum(weights)
        normed = [w / total_w for w in weights]
        return _weighted_median(probs, normed)

    s_con = _consensus(sharp_lines)
    r_con = _consensus(retail_lines)

    result["sharp_consensus"] = round(s_con, 4)
    result["retail_consensus"] = round(r_con, 4)
    result["divergence"] = round(abs(s_con - r_con), 4)

    return result


# ------------------------------------------------------------------
# Hold spread
# ------------------------------------------------------------------

def hold_spread(book_holds: dict[str, float]) -> float | None:
    """Compute IQR (p75 - p25) of book hold percentages.

    Returns None if fewer than 4 books.
    """
    vals = list(book_holds.values())
    if len(vals) < 4:
        return None
    q1, _, q3 = quantiles(vals, n=4)
    return round(q3 - q1, 2)


# ------------------------------------------------------------------
# Interpretation tag
# ------------------------------------------------------------------

_EFFICIENT_HOLD_MAX = 5.0
_EFFICIENT_SIGMA_MAX = 0.03
_EFFICIENT_DIVERGENCE_MAX = 0.03
_NOISY_SIGMA_MIN = 0.06
_NOISY_DIVERGENCE_MIN = 0.05


def market_tag(
    *,
    market_hold_median: float,
    volatility_sigma: float,
    divergence: float | None,
) -> str:
    """Return an interpretation tag for the market.

    "Efficient" — low hold, low sigma, low divergence.
    "Noisy"     — high sigma or high divergence.
    Otherwise   — "Normal".
    """
    div = divergence if divergence is not None else 0.0

    if (
        volatility_sigma >= _NOISY_SIGMA_MIN
        or div >= _NOISY_DIVERGENCE_MIN
    ):
        return "Noisy"

    if (
        market_hold_median <= _EFFICIENT_HOLD_MAX
        and volatility_sigma <= _EFFICIENT_SIGMA_MAX
        and div <= _EFFICIENT_DIVERGENCE_MAX
    ):
        return "Efficient"

    return "Normal"


# ------------------------------------------------------------------
# Full analysis for one market
# ------------------------------------------------------------------

def analyze_market(
    lines: list[BettingLine],
) -> dict | None:
    """Run full market-structure analysis for a set of same-market lines.

    Returns a dict with all metrics, or None if insufficient data.
    """
    if len(lines) < 2:
        return None

    bt = lines[0].bet_type
    books_total = len(lines)

    # Book holds
    book_holds_map: dict[str, float] = {}
    for ln in lines:
        if bt == BetType.MONEYLINE:
            from line_tracker.bet_slip import implied_prob_from_american
            pa = implied_prob_from_american(ln.home_value)
            pb = implied_prob_from_american(ln.away_value)
        else:
            if ln.home_price is None or ln.away_price is None:
                continue
            from line_tracker.bet_slip import implied_prob_from_american
            pa = implied_prob_from_american(ln.home_price)
            pb = implied_prob_from_american(ln.away_price)
        book_holds_map[ln.sportsbook] = round((pa + pb - 1) * 100, 2)

    hold_vals = list(book_holds_map.values())
    mkt_hold_med = round(median(hold_vals), 2) if hold_vals else 0.0
    h_spread = hold_spread(book_holds_map)

    # Volatility sigma (de-vigged home/over side)
    probs = []
    for ln in lines:
        if bt == BetType.MONEYLINE:
            p, _ = _remove_vig(ln.home_value, ln.away_value)
        else:
            if ln.home_price is None or ln.away_price is None:
                continue
            p, _ = _remove_vig(ln.home_price, ln.away_price)
        probs.append(p)

    vol_sigma = round(stdev(probs), 4) if len(probs) >= 2 else 0.0

    # Sharp vs retail
    div_info = sharp_retail_divergence(lines)

    # Line dispersion (spread/total only)
    dispersion = None
    if bt in (BetType.SPREAD, BetType.TOTAL):
        dispersion = line_dispersion(lines)

    tag = market_tag(
        market_hold_median=mkt_hold_med,
        volatility_sigma=vol_sigma,
        divergence=div_info["divergence"],
    )

    return {
        "bet_type": bt.value,
        "books_total": books_total,
        "books_with_holds": len(book_holds_map),
        "book_holds": book_holds_map,
        "market_hold_median": mkt_hold_med,
        "hold_spread": h_spread,
        "volatility_sigma": vol_sigma,
        "sharp_consensus": div_info["sharp_consensus"],
        "retail_consensus": div_info["retail_consensus"],
        "divergence": div_info["divergence"],
        "sharp_books": div_info["sharp_books"],
        "retail_books": div_info["retail_books"],
        "line_dispersion": dispersion,
        "tag": tag,
    }

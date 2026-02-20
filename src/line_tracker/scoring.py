"""Shared scoring and ranking module.

Provides a single, deterministic source of truth for:
- Alpha field computation (delegates to ``alpha.py``)
- Hybrid risk-adjusted score computation
- Market weighting layer (reliability bias for tiering + hybrid ranking)
- Effective Kelly sizing (confidence-scaled bankroll fraction)
- Candidate ranking with pluggable modes

No Streamlit imports, no DB, no global state.
"""

from __future__ import annotations

import os

from line_tracker.alpha import alpha_label, alpha_score
from line_tracker.config import (
    get_conf_w_high,
    get_conf_w_low,
    get_conf_w_max,
    get_conf_w_med,
    get_conf_w_min,
    get_kelly_mult_high,
    get_kelly_mult_low,
    get_kelly_mult_med,
    get_market_weight_longshot,
    get_market_weight_ml_dog,
    get_market_weight_ml_fav,
    get_market_weight_spread,
    get_market_weight_total,
    get_pro_hybrid_market_conf,
)

# ── Hybrid score weights ──────────────────────────────────────────────
_W_ALPHA = 0.40
_W_KELLY = 0.40
_W_PROB = 0.20
_KELLY_CAP = 0.05  # normalise kelly_suggested to this cap

# ── Market weight defaults (overridable via env/secrets) ─────────────
_MW_LONGSHOT_PROB_THRESHOLD = 0.20
_MW_FAVORITE_PROB_THRESHOLD = 0.55

# ── Kelly effective cap (same as kelly_base cap in core/math.py) ─────
_KELLY_EFFECTIVE_CAP = 0.25


def compute_alpha_fields(entry: dict) -> dict:
    """Compute alpha score, label, and components for *entry*.

    Reads fields from *entry* (books_used, edge_z, …) via
    ``alpha.alpha_score`` and returns a dict with keys:
        alpha_score, alpha_label, alpha_components

    Pure function – does **not** mutate *entry*.
    """
    a_score, a_components = alpha_score(entry)
    a_label = alpha_label(a_score)
    return {
        "alpha_score": a_score,
        "alpha_label": a_label,
        "alpha_components": a_components,
    }


# ── Market weighting layer ────────────────────────────────────────────


def _getval(obj: object, key: str, default: object = None) -> object:
    """Read *key* from a dict or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def classify_market(entry: object) -> str:
    """Classify *entry* into a market-weight category.

    Returns one of ``"spreads"``, ``"totals"``, ``"ml_favorite"``,
    ``"ml_underdog"``, or ``"unknown"``.

    Works with both dict entries and ``BetRecommendation`` objects.
    """
    market = str(_getval(entry, "market", "") or "").lower()

    if market in ("spread", "spreads"):
        return "spreads"
    if market in ("total", "totals"):
        return "totals"
    if market in ("moneyline", "h2h"):
        odds = _getval(entry, "best_odds", None)
        prob = _getval(entry, "consensus_prob", None)
        if _is_favorite(odds, prob):
            return "ml_favorite"
        return "ml_underdog"
    return "unknown"


def _is_favorite(odds: float | None, prob: float | None) -> bool:
    """Return True if odds/prob indicate a favorite."""
    if odds is not None:
        try:
            odds_f = float(odds)
            if odds_f < 0:
                return True
            # Positive American odds → decimal >= 2.0 → underdog
        except (TypeError, ValueError):
            pass
    if prob is not None:
        try:
            if float(prob) >= _MW_FAVORITE_PROB_THRESHOLD:
                return True
        except (TypeError, ValueError):
            pass
    return False


def market_weight(entry: object) -> float:
    """Compute the market reliability weight for *entry*.

    Soft bias toward more reliable market types.  Does NOT remove
    candidates — just scales tier_score and hybrid_score.

    Works with both dict entries and ``BetRecommendation`` objects.
    """
    cat = classify_market(entry)

    weight_map = {
        "spreads": get_market_weight_spread(),
        "totals": get_market_weight_total(),
        "ml_favorite": get_market_weight_ml_fav(),
        "ml_underdog": get_market_weight_ml_dog(),
        "unknown": 1.0,
    }
    w = weight_map.get(cat, 1.0)

    # Longshot penalty: ML + consensus_prob < threshold
    if cat in ("ml_favorite", "ml_underdog"):
        prob = _getval(entry, "consensus_prob", 1.0)
        try:
            if float(prob or 1.0) < _MW_LONGSHOT_PROB_THRESHOLD:
                w *= get_market_weight_longshot()
        except (TypeError, ValueError):
            pass

    return round(w, 6)


# ── Confidence-adjusted hybrid market weight ──────────────────────────


def confidence_factor(confidence_label: str) -> float:
    """Return the confidence weight factor for *confidence_label*.

    Maps ``"High"`` / ``"Medium"`` / ``"Low"`` to configurable multipliers.
    Unknown labels default to the Medium factor.
    """
    factor_map = {
        "High": get_conf_w_high(),
        "Medium": get_conf_w_med(),
        "Low": get_conf_w_low(),
    }
    return factor_map.get(confidence_label, get_conf_w_med())


def hybrid_market_weight(entry: object) -> float:
    """Compute the effective market weight for hybrid scoring.

    When ``PRO_HYBRID_MARKET_CONF`` is **off** (default), returns the
    base ``market_weight(entry)`` unchanged.

    When the flag is **on**, the base weight is scaled by the confidence
    factor for the entry's ``confidence_label`` and clamped to
    ``[CONF_W_MIN, CONF_W_MAX]``.
    """
    base = market_weight(entry)
    if not get_pro_hybrid_market_conf():
        return base
    label = str(_getval(entry, "confidence_label", "Medium") or "Medium")
    cf = confidence_factor(label)
    effective = base * cf
    effective = max(get_conf_w_min(), min(effective, get_conf_w_max()))
    return round(effective, 6)


# ── Hybrid score computation ─────────────────────────────────────────


def compute_hybrid_fields(entry: dict) -> dict:
    """Compute the hybrid risk-adjusted ranking score for *entry*.

    Raw formula (unchanged):
        alpha_norm   = alpha_score / 100
        kelly_norm   = clamp(kelly_suggested / 0.05, 0, 1)
        prob_norm    = consensus_prob

        hybrid_score_raw = 0.40 * alpha_norm
                         + 0.40 * kelly_norm
                         + 0.20 * prob_norm

    Market weighting (new):
        hybrid_score = hybrid_score_raw * market_weight(entry)

    Returns a dict with keys:
        hybrid_score, hybrid_score_raw, market_weight, hybrid_components

    Pure function – does **not** mutate *entry*.
    """
    alpha = entry.get("alpha_score") or 0
    kelly = entry.get("kelly_suggested") or 0
    prob = entry.get("consensus_prob") or 0

    alpha_norm = alpha / 100.0
    kelly_norm = max(0.0, min(kelly / _KELLY_CAP, 1.0))
    prob_norm = prob

    raw = round(
        _W_ALPHA * alpha_norm + _W_KELLY * kelly_norm + _W_PROB * prob_norm,
        4,
    )
    base_mw = market_weight(entry)
    eff_mw = hybrid_market_weight(entry)
    score = round(raw * eff_mw, 4)

    return {
        "hybrid_score_raw": raw,
        "hybrid_score": score,
        "market_weight": base_mw,
        "effective_market_weight": eff_mw,
        "hybrid_components": {
            "alpha_norm": round(alpha_norm, 4),
            "kelly_norm": round(kelly_norm, 4),
            "prob_norm": round(prob_norm, 4),
            "market_weight": base_mw,
            "effective_market_weight": eff_mw,
            "weights": {
                "alpha": _W_ALPHA,
                "kelly": _W_KELLY,
                "prob": _W_PROB,
            },
        },
    }


# ── Ranking modes ─────────────────────────────────────────────────────

RANKING_MODES = ("hybrid", "hit", "value")

# Longshot guard for "hit" mode: bets where consensus_prob is below this
# floor are considered longshots and are demoted unless the alpha_label is
# "Strong" (indicating the edge is robust despite the low probability).
LONGSHOT_PROB_FLOOR = 0.20

# Hit-mode top-pick probability floor: candidates below this threshold are
# demoted (but not removed) so they don't appear as the #1 pick.
PROB_FLOOR_HIT = 0.30


def _hybrid_sort_key(e: dict) -> tuple:
    """Primary: hybrid_score (desc), then edge_ev_shrunk, quality_score."""
    return (
        -e.get("hybrid_score", 0.0),
        -e.get("edge_ev_shrunk", 0.0),
        -e.get("quality_score", 0),
    )


def _hit_sort_key(e: dict) -> tuple:
    """Probability-first ranking for hit mode.

    consensus_prob is the dominant ordering.  The only hard partition is
    the PROB_FLOOR_HIT gate — candidates at or above 0.30 always rank
    above those below, regardless of edge or alpha.

    Order:
    1. Above PROB_FLOOR_HIT (0.30) first — hard partition so longshots
       never leapfrog solid favorites.
    2. Higher consensus_prob first (dominant sort key).
    3. Higher alpha_score as tiebreaker.
    4. Higher quality_score as tiebreaker.
    5. Higher agreement_score (tighter book consensus).
    6. Lower sigma (less market noise).
    7. Higher edge as LAST tiebreaker (never ahead of prob).
    """
    prob = e.get("consensus_prob", 0.0)
    above_floor = 1 if prob >= PROB_FLOOR_HIT else 0
    return (
        -above_floor,                                # 1. above prob floor first
        -prob,                                       # 2. highest prob first (dominant)
        -e.get("alpha_score", 0),                    # 3. highest alpha
        -e.get("quality_score", 0),                  # 4. highest quality
        -e.get("agreement_score", 0.0),              # 5. highest agreement
        e.get("market_volatility_sigma", 0.0),       # 6. lowest sigma
        -e.get("edge_ev_shrunk", 0.0),               # 7. edge LAST
    )


def _value_sort_key(e: dict) -> tuple:
    """EV-first ranking (matches legacy best-bet ordering).

    Order: edge_ev_shrunk desc, then ev_100, quality_score.
    """
    return (
        -e.get("edge_ev_shrunk", 0.0),
        -e.get("ev_100", 0.0),
        -e.get("quality_score", 0),
    )


_MODE_KEY = {
    "hybrid": _hybrid_sort_key,
    "hit": _hit_sort_key,
    "value": _value_sort_key,
}


def rank_candidates(
    candidates: list[dict],
    mode: str = "hybrid",
) -> list[dict]:
    """Sort *candidates* in-place and return the sorted list.

    Parameters
    ----------
    candidates:
        List of entry dicts.  Each must already have alpha/hybrid fields
        populated (call ``enrich_entry`` or compute functions first).
    mode:
        One of ``"hybrid"`` (default), ``"hit"``, or ``"value"``.

    Returns
    -------
    The same list, sorted according to *mode*.
    """
    key_fn = _MODE_KEY.get(mode)
    if key_fn is None:
        raise ValueError(
            f"Unknown ranking mode {mode!r}; choose from {RANKING_MODES}"
        )
    candidates.sort(key=key_fn)
    _debug_ranking(candidates, mode)
    return candidates


# ── Mode-aware filtering ──────────────────────────────────────────────


def filter_candidates(
    ranked: list[dict],
    mode: str,
    min_edge: float,
    min_quality: int,
) -> list[dict]:
    """Apply mode-aware eligibility filters to *ranked* candidates.

    For ``"hit"`` mode the edge gate is skipped (edge is a tiebreaker
    in the sort key, not a gate).  Only min_quality is enforced.

    For ``"hybrid"`` and ``"value"`` modes both min_edge and min_quality
    gates apply (existing behaviour).

    Returns a new list; does not mutate the input.
    """
    if mode == "hit":
        return [
            e for e in ranked
            if e.get("quality_score", 0) >= min_quality
        ]
    # hybrid / value — apply both gates
    return [
        e for e in ranked
        if e.get("edge_shrunk_pct", e.get("edge_pct", 0.0)) >= min_edge
        and e.get("quality_score", 0) >= min_quality
    ]


# ── Debug output ─────────────────────────────────────────────────────


def _debug_ranking(candidates: list[dict], mode: str) -> None:
    """Print top-5 candidates when DEBUG_RANKING=1."""
    if not os.environ.get("DEBUG_RANKING"):
        return
    print(f"\n[DEBUG_RANKING] mode={mode}  top-5:")
    for i, e in enumerate(candidates[:5]):
        print(
            f"  {i + 1}. {e.get('selection', '?')}/{e.get('market', '?')} "
            f"prob={e.get('consensus_prob', 0):.3f} "
            f"edge={e.get('edge_shrunk_pct', 0):.2f}% "
            f"quality={e.get('quality_score', 0)} "
            f"alpha={e.get('alpha_label', '?')} "
            f"hybrid={e.get('hybrid_score', 0):.3f}"
        )


# ── Convenience: enrich a single entry with all scoring fields ────────


def compute_confidence_label(entry: dict) -> str:
    """Derive a confidence label from existing signals.

    This is a **gating-only** label used for Tier 1 eligibility and
    primary-card selection.  It does NOT affect EV math, ranking order,
    or the existing ``confidence`` field (edge-z based).

    Inputs (all read from *entry*):
        consensus_prob, quality_score, market_hold_median (as hold_max),
        alpha_label, edge_z (optional).

    Returns ``"High"``, ``"Medium"``, or ``"Low"``.
    """
    quality_score = entry.get("quality_score", 0)
    consensus_prob = entry.get("consensus_prob", 0.0)
    hold_max = entry.get("market_hold_median", 0.0) / 100.0  # pct → decimal
    alpha_label_val = entry.get("alpha_label", "")
    edge_z = entry.get("edge_z", 0.0)

    # HIGH: quality >= 70, prob >= 0.45, hold <= 0.09,
    #       AND (alpha Strong OR edge_z >= 0.8)
    if (
        quality_score >= 70
        and consensus_prob >= 0.45
        and hold_max <= 0.09
        and (alpha_label_val == "Strong" or edge_z >= 0.8)
    ):
        return "High"

    # MEDIUM: quality >= 60, prob >= 0.35, hold <= 0.10
    if (
        quality_score >= 60
        and consensus_prob >= 0.35
        and hold_max <= 0.10
    ):
        return "Medium"

    # LOW: everything else
    return "Low"


# ── Effective Kelly sizing ────────────────────────────────────────────


def compute_kelly_effective(entry: dict) -> dict:
    """Compute effective Kelly fields from kelly_base and confidence_label.

    kelly_effective = kelly_base * confidence_label_multiplier

    The multiplier is based on the *confidence_label* (quality/prob/hold/alpha),
    NOT the edge-z ``confidence`` field.

    Returns a dict with keys:
        kelly_raw, kelly_multiplier, kelly_effective
    """
    kelly_raw = entry.get("kelly_base", 0.0) or 0.0
    conf_label = entry.get("confidence_label", "Low")

    mult_map = {
        "High": get_kelly_mult_high(),
        "Medium": get_kelly_mult_med(),
        "Low": get_kelly_mult_low(),
    }
    mult = mult_map.get(conf_label, get_kelly_mult_low())
    kelly_eff = round(kelly_raw * mult, 6)
    # Clamp to same bounds as kelly_base
    kelly_eff = max(0.0, min(kelly_eff, _KELLY_EFFECTIVE_CAP))

    return {
        "kelly_raw": round(kelly_raw, 6),
        "kelly_multiplier": mult,
        "kelly_effective": kelly_eff,
    }


# ── Convenience: enrich a single entry with all scoring fields ────────


def enrich_entry(entry: dict) -> dict:
    """Add alpha, hybrid, confidence-label, and kelly-effective fields (mutates).

    Convenience wrapper that calls ``compute_alpha_fields``,
    ``compute_hybrid_fields``, ``compute_confidence_label``, and
    ``compute_kelly_effective`` and merges results into *entry*.

    Returns *entry* for chaining.
    """
    entry.update(compute_alpha_fields(entry))
    entry.update(compute_hybrid_fields(entry))
    entry["confidence_label"] = compute_confidence_label(entry)
    entry.update(compute_kelly_effective(entry))
    return entry

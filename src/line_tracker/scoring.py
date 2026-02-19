"""Shared scoring and ranking module.

Provides a single, deterministic source of truth for:
- Alpha field computation (delegates to ``alpha.py``)
- Hybrid risk-adjusted score computation
- Candidate ranking with pluggable modes

No Streamlit imports, no DB, no global state.
"""

from __future__ import annotations

import os

from line_tracker.alpha import alpha_label, alpha_score

# ── Hybrid score weights ──────────────────────────────────────────────
_W_ALPHA = 0.40
_W_KELLY = 0.40
_W_PROB = 0.20
_KELLY_CAP = 0.05  # normalise kelly_suggested to this cap


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


def compute_hybrid_fields(entry: dict) -> dict:
    """Compute the hybrid risk-adjusted ranking score for *entry*.

    Formula (unchanged from original ``slate.compute_hybrid_score``):
        alpha_norm   = alpha_score / 100
        kelly_norm   = clamp(kelly_suggested / 0.05, 0, 1)
        prob_norm    = consensus_prob

        hybrid_score = 0.40 * alpha_norm
                     + 0.40 * kelly_norm
                     + 0.20 * prob_norm

    Returns a dict with keys:
        hybrid_score, hybrid_components

    Pure function – does **not** mutate *entry*.
    """
    alpha = entry.get("alpha_score") or 0
    kelly = entry.get("kelly_suggested") or 0
    prob = entry.get("consensus_prob") or 0

    alpha_norm = alpha / 100.0
    kelly_norm = max(0.0, min(kelly / _KELLY_CAP, 1.0))
    prob_norm = prob

    score = round(
        _W_ALPHA * alpha_norm + _W_KELLY * kelly_norm + _W_PROB * prob_norm,
        4,
    )
    return {
        "hybrid_score": score,
        "hybrid_components": {
            "alpha_norm": round(alpha_norm, 4),
            "kelly_norm": round(kelly_norm, 4),
            "prob_norm": round(prob_norm, 4),
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


def enrich_entry(entry: dict) -> dict:
    """Add alpha and hybrid fields to *entry* (mutates in-place).

    Convenience wrapper that calls ``compute_alpha_fields`` and
    ``compute_hybrid_fields`` and merges results into *entry*.

    Returns *entry* for chaining.
    """
    entry.update(compute_alpha_fields(entry))
    entry.update(compute_hybrid_fields(entry))
    return entry

"""Shared scoring and ranking module.

Provides a single, deterministic source of truth for:
- Alpha field computation (delegates to ``alpha.py``)
- Hybrid risk-adjusted score computation
- Candidate ranking with pluggable modes

No Streamlit imports, no DB, no global state.
"""

from __future__ import annotations

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


def _hybrid_sort_key(e: dict) -> tuple:
    """Primary: hybrid_score (desc), then edge_ev_shrunk, quality_score."""
    return (
        -e.get("hybrid_score", 0.0),
        -e.get("edge_ev_shrunk", 0.0),
        -e.get("quality_score", 0),
    )


def _hit_sort_key(e: dict) -> tuple:
    """Prioritise most-likely-to-hit while keeping basic sanity.

    Order:
    1. Require non-negative shrunk edge (demote negatives to bottom).
    2. Higher consensus_prob first.
    3. Higher alpha_score (penalises longshots via alpha penalties).
    4. Higher agreement_score (tighter book consensus).
    5. Lower sigma (less market noise).
    """
    edge_ok = 1 if e.get("edge_ev_shrunk", 0.0) >= 0 else 0
    return (
        -edge_ok,                                    # non-neg edge first
        -e.get("consensus_prob", 0.0),               # highest prob first
        -e.get("alpha_score", 0),                    # highest alpha first
        -e.get("agreement_score", 0.0),              # highest agreement first
        e.get("market_volatility_sigma", 0.0),       # lowest sigma first
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
    return candidates


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

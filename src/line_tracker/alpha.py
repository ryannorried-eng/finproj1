"""Alpha v1.1 – "More Selective" interpretable robustness overlay.

Scores each recommendation 0–100 using existing computed fields plus
optional best-odds and market-type inputs.  Uses smooth (smoothstep)
scoring curves instead of hard tier buckets, and adds implied-odds
longshot penalty + moneyline-dog adjustment.

No ML, no external APIs – pure rule-based scoring from market structure,
edge statistics, book agreement, consensus probability, and odds.
"""

from __future__ import annotations

# ── Alpha gating toggle ──────────────────────────────────────────────
# When True, tier1b recs with a "Weak" alpha label are downgraded to
# tier2.  Default: False (shadow mode – scores computed but no gating).
ALPHA_GATE_ENABLED = False


# ── Smooth-scoring helpers ───────────────────────────────────────────


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp *x* into [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from *a* to *b* by factor *t*."""
    return a + (b - a) * t


def smoothstep(lo: float, hi: float, x: float) -> float:
    """Hermite smoothstep: returns 0..1 with smooth S-curve.

    * x <= lo  → 0
    * x >= hi  → 1
    * in between → t²(3 − 2t)  where t = (x−lo)/(hi−lo)
    """
    if hi == lo:
        return 0.0 if x < lo else 1.0
    t = clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ── Implied-probability helpers ──────────────────────────────────────


def implied_prob_from_american(odds: float) -> float | None:
    """Convert American odds to implied probability.

    Returns None if odds are in the dead-zone (-100, 100) exclusive.
    """
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    if odds <= -100:
        return (-odds) / ((-odds) + 100.0)
    return None


def _implied_longshot_penalty(implied_prob: float | None) -> float:
    """Penalty based on implied probability from best odds (0 to −12)."""
    if implied_prob is None:
        return 0.0
    if implied_prob >= 0.25:
        return 0.0
    if implied_prob >= 0.20:
        return -2.0
    if implied_prob >= 0.15:
        return -5.0
    if implied_prob >= 0.10:
        return -9.0
    return -12.0


def _ml_penalty(
    market_type: str | None,
    implied_prob: float | None,
) -> float:
    """Moneyline / h2h market-type penalty (0 to −7)."""
    if market_type is None:
        return 0.0
    mt = market_type.lower()
    if mt not in ("moneyline", "h2h"):
        return 0.0
    base = -4.0
    if implied_prob is not None and implied_prob < 0.20:
        base = -7.0
    return base


# ── Main scoring function ────────────────────────────────────────────


def alpha_score(
    entry: dict,
    *,
    best_odds_american: float | None = None,
    market_type: str | None = None,
) -> tuple[int, dict]:
    """Score a recommendation entry for robustness (0–100).

    Parameters
    ----------
    entry:
        A slate entry dict containing at least: books_used,
        market_hold_median, edge_z, edge_ev_shrunk, agreement_score,
        market_volatility_sigma, consensus_prob.
        May also contain best_odds and market for fallback values.
    best_odds_american:
        Best available American odds (optional; falls back to
        entry["best_odds"]).
    market_type:
        Market type string e.g. "moneyline", "spread", "total"
        (optional; falls back to entry["market"]).

    Returns
    -------
    (score, components) where score is clamped to [0, 100] and
    components is a dict of each sub-score plus the inputs used.
    """
    books = entry.get("books_used", 0)
    hold = entry.get("market_hold_median", 0.0)
    edge_z = entry.get("edge_z", 0.0)
    edge_ev_shrunk = entry.get("edge_ev_shrunk", 0.0)
    agreement = entry.get("agreement_score", 0.0)
    sigma = entry.get("market_volatility_sigma", 0.0)
    consensus_prob = entry.get("consensus_prob", 0.0)

    # Resolve optional inputs with fallbacks from entry dict
    if best_odds_american is None:
        best_odds_american = entry.get("best_odds")
    if market_type is None:
        market_type = entry.get("market") or entry.get("market_type")

    # ── A) Market robustness (0–30) ──────────────────────────────────
    score_books = 18.0 * smoothstep(4.0, 10.0, float(books))
    score_hold = 12.0 * (1.0 - smoothstep(4.5, 8.0, hold))
    books_pts = round(score_books, 4)
    hold_pts = round(score_hold, 4)
    market_robustness = round(clamp(books_pts + hold_pts, 0.0, 30.0), 4)

    # ── B) Edge robustness (0–40) ────────────────────────────────────
    score_z = 20.0 * smoothstep(0.75, 2.25, edge_z)
    shrunk_pct = edge_ev_shrunk * 100.0
    score_shrunk = 20.0 * smoothstep(0.50, 3.00, shrunk_pct)
    edge_z_pts = round(score_z, 4)
    shrunk_pts = round(score_shrunk, 4)
    edge_robustness = round(edge_z_pts + shrunk_pts, 4)

    # ── C) Agreement / stability (0–20) ──────────────────────────────
    score_agree = 12.0 * smoothstep(70.0, 95.0, agreement)
    score_sigma = 8.0 * (1.0 - smoothstep(0.006, 0.016, sigma))
    agree_pts = round(score_agree, 4)
    sigma_pts = round(score_sigma, 4)
    agreement_stability = round(clamp(agree_pts + sigma_pts, 0.0, 20.0), 4)

    # ── D) Consensus-prob longshot penalty (0 to −18) ────────────────
    t_cons = smoothstep(0.18, 0.30, consensus_prob)
    penalty_consensus = round(-18.0 * (1.0 - t_cons), 4)

    # ── E) Implied-odds longshot penalty (0 to −12) ──────────────────
    implied_prob: float | None = None
    if best_odds_american is not None:
        implied_prob = implied_prob_from_american(best_odds_american)
    penalty_implied = _implied_longshot_penalty(implied_prob)

    # ── F) Market-type ML penalty (0 to −7) ──────────────────────────
    ml_pen = _ml_penalty(market_type, implied_prob)

    # ── Final score ──────────────────────────────────────────────────
    raw = (
        market_robustness
        + edge_robustness
        + agreement_stability
        + penalty_consensus
        + penalty_implied
        + ml_pen
    )
    score = round(clamp(raw, 0.0, 100.0))

    components = {
        "market_robustness": market_robustness,
        "books_pts": books_pts,
        "hold_pts": hold_pts,
        "edge_robustness": edge_robustness,
        "edge_z_pts": edge_z_pts,
        "shrunk_pts": shrunk_pts,
        "agreement_stability": agreement_stability,
        "agree_pts": agree_pts,
        "sigma_pts": sigma_pts,
        "longshot_penalty_consensus": penalty_consensus,
        "longshot_penalty_implied": penalty_implied,
        "ml_penalty": ml_pen,
        "raw_total": round(raw, 4),
        "final_score": score,
        "inputs": {
            "books_used": books,
            "market_hold_median": hold,
            "edge_z": edge_z,
            "edge_ev_shrunk": edge_ev_shrunk,
            "shrunk_pct": round(shrunk_pct, 4),
            "agreement_score": agreement,
            "market_volatility_sigma": sigma,
            "consensus_prob": consensus_prob,
            "implied_prob": (
                round(implied_prob, 6) if implied_prob is not None
                else None
            ),
            "market_type": market_type,
        },
    }

    return score, components


def alpha_label(score: int) -> str:
    """Map an alpha score to a human-readable label.

    Returns
    -------
    "Strong" if score >= 72, "Neutral" if score >= 45, else "Weak".
    """
    if score >= 72:
        return "Strong"
    if score >= 45:
        return "Neutral"
    return "Weak"

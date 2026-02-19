"""Alpha v1 – interpretable robustness overlay for slate recommendations.

Scores each recommendation 0–100 using existing computed fields.
No ML, no external APIs – pure rule-based scoring from market structure,
edge statistics, book agreement, and consensus probability.
"""

from __future__ import annotations

# ── Alpha gating toggle ──────────────────────────────────────────────
# When True, tier1b recs with a "Weak" alpha label are downgraded to
# tier2.  Default: False (shadow mode – scores computed but no gating).
ALPHA_GATE_ENABLED = False


def alpha_score(entry: dict) -> tuple[int, dict]:
    """Score a recommendation entry for robustness (0–100).

    Parameters
    ----------
    entry:
        A slate entry dict containing at least: books_used,
        market_hold_median, edge_z, edge_ev_shrunk, agreement_score,
        market_volatility_sigma, consensus_prob.

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

    # ── A) Market robustness (0–30) ──────────────────────────────────
    if books >= 7:
        books_pts = 15
    elif books >= 5:
        books_pts = 10
    elif books >= 4:
        books_pts = 5
    else:
        books_pts = 0

    if hold <= 5.5:
        hold_pts = 15
    elif hold <= 7.5:
        hold_pts = 10
    elif hold <= 8.5:
        hold_pts = 5
    else:
        hold_pts = 0

    market_robustness = books_pts + hold_pts

    # ── B) Edge robustness (0–40) ────────────────────────────────────
    if edge_z >= 2.5:
        ez_pts = 20
    elif edge_z >= 2.0:
        ez_pts = 15
    elif edge_z >= 1.75:
        ez_pts = 10
    elif edge_z >= 1.4:
        ez_pts = 5
    else:
        ez_pts = 0

    # Convert edge_ev_shrunk (prob space) to EV/$100 for thresholds
    shrunk_pct = edge_ev_shrunk * 100.0
    if shrunk_pct >= 1.25:
        shrunk_pts = 20
    elif shrunk_pct >= 0.75:
        shrunk_pts = 15
    elif shrunk_pct >= 0.35:
        shrunk_pts = 10
    elif shrunk_pct > 0:
        shrunk_pts = 5
    else:
        shrunk_pts = 0

    edge_robustness = ez_pts + shrunk_pts

    # ── C) Agreement / stability (0–20) ──────────────────────────────
    if agreement >= 90:
        agree_pts = 10
    elif agreement >= 80:
        agree_pts = 7
    elif agreement >= 70:
        agree_pts = 4
    else:
        agree_pts = 0

    if sigma <= 0.0045:
        sigma_pts = 10
    elif sigma <= 0.007:
        sigma_pts = 7
    elif sigma <= 0.010:
        sigma_pts = 4
    else:
        sigma_pts = 0

    agreement_stability = agree_pts + sigma_pts

    # ── D) Longshot penalty (0 to -25) ───────────────────────────────
    if consensus_prob >= 0.45:
        penalty = 0
    elif consensus_prob >= 0.30:
        penalty = -5
    elif consensus_prob >= 0.20:
        penalty = -12
    elif consensus_prob >= 0.15:
        penalty = -18
    else:
        penalty = -25

    # ── Final score ──────────────────────────────────────────────────
    raw = market_robustness + edge_robustness + agreement_stability + penalty
    score = max(0, min(100, raw))

    components = {
        "market_robustness": market_robustness,
        "books_pts": books_pts,
        "hold_pts": hold_pts,
        "edge_robustness": edge_robustness,
        "edge_z_pts": ez_pts,
        "shrunk_pts": shrunk_pts,
        "agreement_stability": agreement_stability,
        "agree_pts": agree_pts,
        "sigma_pts": sigma_pts,
        "longshot_penalty": penalty,
        "raw_total": raw,
        # Inputs used
        "inputs": {
            "books_used": books,
            "market_hold_median": hold,
            "edge_z": edge_z,
            "edge_ev_shrunk": edge_ev_shrunk,
            "shrunk_pct": round(shrunk_pct, 4),
            "agreement_score": agreement,
            "market_volatility_sigma": sigma,
            "consensus_prob": consensus_prob,
        },
    }

    return score, components


def alpha_label(score: int) -> str:
    """Map an alpha score to a human-readable label.

    Returns
    -------
    "Strong" if score >= 65, "Neutral" if score >= 40, else "Weak".
    """
    if score >= 65:
        return "Strong"
    if score >= 40:
        return "Neutral"
    return "Weak"

"""Step 7 – "Market + behavior" model to predict CLV > 0 (NOT game outcome).

Trains a simple interpretable model from closed rec_snapshots.  The model
learns which (market, selection_type) × feature combinations historically
produce positive CLV.

No external ML libraries — pure Python weighted-average scoring.
"""

from __future__ import annotations

import math
from statistics import mean


def extract_features(snap: dict) -> dict:
    """Extract model features from a closed rec_snapshot row.

    Features are market-structure and behavior signals, NOT game outcome
    predictors.
    """
    market = snap.get("market", "")
    selection = snap.get("selection", "")
    consensus = snap.get("consensus_prob") or 0.0

    # Heuristic selection_type classification
    selection_type = _classify_selection(market, selection, consensus)

    return {
        "market": market,
        "selection_type": selection_type,
        "edge_z": snap.get("edge_z") or 0.0,
        "edge_ev_shrunk": snap.get("edge_ev_shrunk") or 0.0,
        "quality_score": snap.get("quality_score") or 0,
        "consensus_prob": consensus,
        "alpha_score": snap.get("alpha_score") or 0,
    }


def _classify_selection(
    market: str, selection: str, consensus_prob: float,
) -> str:
    """Heuristic: classify a selection into a behavioral type."""
    m = market.lower()
    if m in ("total", "totals"):
        s = selection.lower()
        if "over" in s:
            return "over"
        if "under" in s:
            return "under"
        return "total_other"
    if m in ("moneyline", "h2h"):
        if consensus_prob >= 0.55:
            return "ml_favorite"
        if consensus_prob <= 0.45:
            return "ml_underdog"
        return "ml_tossup"
    if m in ("spread", "spreads"):
        if consensus_prob >= 0.55:
            return "spread_favorite"
        return "spread_dog"
    return "other"


def train_clv_model(store) -> dict:
    """Train the CLV-positive prediction model from historical data.

    For each (market, selection_type) group, computes the empirical
    CLV-positive rate and a feature-weighted score.  Stores results
    in the ``clv_model_scores`` table.

    Returns a summary dict.
    """
    closed = store.get_closed_snapshots()
    if not closed:
        return {"status": "no_data", "groups": 0}

    # Extract features + label (clv_positive)
    samples: list[dict] = []
    for snap in closed:
        feat = extract_features(snap)
        clv_positive = (snap.get("clv_delta_implied") or 0) > 0
        feat["clv_positive"] = clv_positive
        samples.append(feat)

    # Group by (market, selection_type)
    groups: dict[tuple, list[dict]] = {}
    for s in samples:
        key = (s["market"], s["selection_type"])
        groups.setdefault(key, []).append(s)

    scores: list[dict] = []
    for (market, sel_type), group_samples in groups.items():
        if len(group_samples) < 3:
            continue

        # Empirical CLV-positive rate
        pos_count = sum(1 for s in group_samples if s["clv_positive"])
        empirical_rate = pos_count / len(group_samples)

        # Feature means (for interpretability)
        avg_edge_z = mean(s["edge_z"] for s in group_samples)
        avg_quality = mean(s["quality_score"] for s in group_samples)
        avg_alpha = mean(s["alpha_score"] for s in group_samples)

        # Weighted composite: blend empirical rate with feature signals
        # Higher edge_z, quality, alpha → higher predicted prob
        feature_signal = _sigmoid(
            0.3 * _z_score_safe(avg_edge_z, 1.5, 0.5)
            + 0.3 * _z_score_safe(avg_quality, 70, 15)
            + 0.2 * _z_score_safe(avg_alpha, 50, 20)
            + 0.2 * _z_score_safe(empirical_rate, 0.5, 0.15)
        )

        # Bayesian blend: 70% empirical, 30% feature signal
        n = len(group_samples)
        weight_empirical = min(n / (n + 10), 0.8)
        predicted = (
            weight_empirical * empirical_rate
            + (1 - weight_empirical) * feature_signal
        )

        scores.append({
            "market": market,
            "selection_type": sel_type,
            "features": {
                "sample_count": n,
                "empirical_clv_positive_rate": round(empirical_rate, 4),
                "avg_edge_z": round(avg_edge_z, 4),
                "avg_quality_score": round(avg_quality, 1),
                "avg_alpha_score": round(avg_alpha, 1),
                "feature_signal": round(feature_signal, 4),
            },
            "predicted_clv_positive_prob": round(predicted, 4),
            "model_version": "v1",
        })

    if scores:
        store.save_clv_model_scores(scores)

    return {"status": "trained", "groups": len(scores), "total_samples": len(samples)}


def predict_clv_positive(entry: dict, store) -> float:
    """Predict the probability of CLV > 0 for a given entry.

    Returns 0.5 (uninformative prior) if no model data exists.
    """
    feat = extract_features(entry)
    score = store.get_clv_model_score(feat["market"], feat["selection_type"])
    if score is None:
        return 0.5
    return score.get("predicted_clv_positive_prob", 0.5)


def refresh_model(store) -> dict:
    """Convenience wrapper: retrain and return summary."""
    return train_clv_model(store)


def _sigmoid(x: float) -> float:
    """Standard sigmoid function, clamped for numerical safety."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _z_score_safe(value: float, center: float, scale: float) -> float:
    """Compute a z-score with a fixed scale (avoids division by zero)."""
    if scale <= 0:
        return 0.0
    return (value - center) / scale

"""Generate predictions for upcoming NCAAB games.

Converts model point-margin predictions into probabilities for moneyline,
spread, and totals markets using the normal distribution (CDF).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from line_tracker.core.logging import get_logger
from line_tracker.model import data, features, train

log = get_logger(__name__)

# Historical standard deviation of NCAAB point margins.
MARGIN_SIGMA = 10.5

# Totals have higher variance than side margins in college basketball.
TOTAL_SIGMA = 12.5


def margin_to_ml_prob(
    predicted_margin: float, sigma: float = MARGIN_SIGMA
) -> float:
    """Convert a predicted home margin into P(home wins).

    Uses the normal CDF:  P(home wins) = Φ(predicted_margin / σ)

    >>> margin_to_ml_prob(0.0)
    0.5
    >>> round(margin_to_ml_prob(3.0), 3)
    0.613
    """
    return float(norm.cdf(predicted_margin / sigma))


def margin_to_spread_prob(
    predicted_margin: float,
    market_spread: float,
    sigma: float = MARGIN_SIGMA,
) -> float:
    """P(home covers) given a market spread.

    ``market_spread`` is negative for a home favourite (e.g. -2.5 means the
    home team is favoured by 2.5 points).

    P(home covers) = Φ((predicted_margin - market_spread) / σ)
    """
    return float(norm.cdf((predicted_margin - market_spread) / sigma))


def margin_to_total_prob(
    predicted_total: float,
    market_total: float,
    sigma: float = TOTAL_SIGMA,
) -> float:
    """P(over) given a market total.

    P(over) = Φ((predicted_total - market_total) / σ)
    """
    return float(norm.cdf((predicted_total - market_total) / sigma))


# ------------------------------------------------------------------
# Game-level prediction
# ------------------------------------------------------------------

def _normalize_for_cmp(name: str) -> str:
    """Normalize for comparison: lowercase, strip periods, ``State`` → ``St``."""
    import re

    s = name.lower().replace(".", "")
    s = re.sub(r"\bstate\b", "st", s)
    return re.sub(r"\s+", " ", s).strip()


def _build_name_index(torvik_names: list[str]) -> dict[str, str]:
    """Build a lookup mapping normalised display names → Torvik names.

    ESPN uses full names like ``"Duke Blue Devils"`` while Torvik uses
    ``"Duke"``.  We build a case-insensitive prefix index so that if the
    ESPN name *starts with* a Torvik name it resolves correctly.
    """
    index: dict[str, str] = {}
    # Exact (lowered) match first
    for name in torvik_names:
        index[name.lower()] = name

    # Add "St." ↔ "State" variants so both spellings resolve correctly.
    for name in torvik_names:
        low = name.lower()
        if " st." in low:
            variant = low.replace(" st.", " state")
            index.setdefault(variant, name)
        elif " state" in low:
            variant = low.replace(" state", " st.")
            index.setdefault(variant, name)

    # Common ESPN→Torvik overrides where prefix matching fails
    _OVERRIDES = {
        "uconn huskies": "Connecticut",
        "hawai'i rainbow warriors": "Hawaii",
        "miami hurricanes": "Miami FL",
        "miami (oh) redhawks": "Miami OH",
        "lsu tigers": "LSU",
        "vcu rams": "VCU",
        "smu mustangs": "SMU",
        "tcu horned frogs": "TCU",
        "byu cougars": "BYU",
        "ucf knights": "UCF",
        "usc trojans": "USC",
        "ole miss rebels": "Ole Miss",
        "umbc retrievers": "UMBC",
        "unc asheville bulldogs": "UNC Asheville",
        "unc greensboro spartans": "UNC Greensboro",
        "unc wilmington seahawks": "UNC Wilmington",
        "utep miners": "UTEP",
        "utsa roadrunners": "UTSA",
        "unlv rebels": "UNLV",
    }
    for espn_lower, torvik in _OVERRIDES.items():
        if torvik in torvik_names or torvik.lower() in index:
            index[espn_lower] = torvik

    return index


def _resolve_team(name: str, name_index: dict[str, str]) -> str | None:
    """Resolve an ESPN display name to a Torvik team name."""
    low = name.lower()

    # Exact match
    if low in name_index:
        return name_index[low]

    # Try comparison-normalised exact match (handles "State" vs "St.")
    low_cmp = _normalize_for_cmp(low)
    for key, torvik in name_index.items():
        if _normalize_for_cmp(key) == low_cmp:
            return torvik

    # Try prefix: longest Torvik name that the ESPN name starts with,
    # requiring a word boundary after the prefix to avoid "Iowa" matching
    # "Iowa State".
    best: str | None = None
    best_len = 0
    for key, torvik in name_index.items():
        key_cmp = _normalize_for_cmp(key)
        if low_cmp.startswith(key_cmp) and len(key_cmp) > best_len:
            # Require word boundary: next char must be space or end-of-string
            if len(low_cmp) == len(key_cmp) or low_cmp[len(key_cmp)] == " ":
                best = torvik
                best_len = len(key_cmp)
    return best


def predict_games(
    games: list[dict],
    *,
    model_path: str | None = None,
) -> list[dict]:
    """Generate margin predictions for a list of upcoming games.

    Parameters
    ----------
    games
        Each dict must contain ``home_team``, ``away_team``, and optionally
        ``neutral_site`` (bool) and ``game_date`` (str).
    model_path
        Explicit path to a trained ``.joblib`` model.  When *None* the most
        recent model is loaded automatically.

    Returns
    -------
    list[dict]
        One dict per game with keys: home_team, away_team, game_date,
        predicted_margin, margin_sigma, home_ml_prob, away_ml_prob,
        model_version, prediction_timestamp.
    """
    model, scaler, metadata = train.load_model(model_path)
    ratings_df = data.fetch_team_ratings()
    ratings_idx = ratings_df.set_index("team")
    name_index = _build_name_index(list(ratings_idx.index))

    timestamp = datetime.now(timezone.utc).isoformat()
    model_version = metadata.get("model_path", "unknown")

    predictions: list[dict] = []
    for game in games:
        home_raw = game["home_team"]
        away_raw = game["away_team"]

        home_name = _resolve_team(home_raw, name_index)
        away_name = _resolve_team(away_raw, name_index)

        if home_name is None or home_name not in ratings_idx.index:
            log.warning("Team not found in ratings: %s – skipping", home_raw)
            continue
        if away_name is None or away_name not in ratings_idx.index:
            log.warning("Team not found in ratings: %s – skipping", away_raw)
            continue

        home_stats = ratings_idx.loc[home_name].to_dict()
        away_stats = ratings_idx.loc[away_name].to_dict()

        neutral = game.get("neutral_site", False)
        feat = features.build_game_features(
            home_stats, away_stats, neutral_site=neutral,
        )
        X = pd.DataFrame([feat])[features.FEATURE_COLUMNS]
        X_scaled = scaler.transform(X)

        margin = float(model.predict(X_scaled)[0])
        home_prob = margin_to_ml_prob(margin)

        predictions.append({
            "home_team": home_name,
            "away_team": away_name,
            "game_date": game.get("game_date", ""),
            "predicted_margin": round(margin, 2),
            "margin_sigma": MARGIN_SIGMA,
            "home_ml_prob": round(home_prob, 4),
            "away_ml_prob": round(1 - home_prob, 4),
            "model_version": model_version,
            "prediction_timestamp": timestamp,
        })

    log.info("Generated %d predictions out of %d games", len(predictions), len(games))
    return predictions


def predict_with_market(
    games: list[dict],
    market_lines: dict,
) -> list[dict]:
    """Predict games and compare against market lines.

    Parameters
    ----------
    games
        Same format as :func:`predict_games`.
    market_lines
        Keyed by a matchup identifier ``"home_team vs away_team"``.  Each
        value is a dict that may contain:

        - ``spread`` – the market spread (negative = home favoured)
        - ``total`` – the market over/under line
        - ``home_ml_odds`` – best available home moneyline (American odds)
        - ``away_ml_odds`` – best available away moneyline (American odds)

    Returns
    -------
    list[dict]
        Same as :func:`predict_games` plus spread / total / edge fields.
    """
    base_preds = predict_games(games)

    enriched: list[dict] = []
    for pred in base_preds:
        key = f"{pred['home_team']} vs {pred['away_team']}"
        line = market_lines.get(key, {})

        margin = pred["predicted_margin"]

        # Spread
        if "spread" in line:
            spread = line["spread"]
            home_sp = margin_to_spread_prob(margin, spread)
            pred["market_spread"] = spread
            pred["home_spread_prob"] = round(home_sp, 4)
            pred["away_spread_prob"] = round(1 - home_sp, 4)
        else:
            pred["market_spread"] = None
            pred["home_spread_prob"] = None
            pred["away_spread_prob"] = None

        # Total
        if "total" in line:
            total_line = line["total"]
            # Use tempo-based predicted total if available; fall back to a
            # simple heuristic: average of market total and a default.
            predicted_total = line.get("predicted_total", total_line)
            over_p = margin_to_total_prob(predicted_total, total_line)
            pred["market_total"] = total_line
            pred["over_prob"] = round(over_p, 4)
            pred["under_prob"] = round(1 - over_p, 4)
        else:
            pred["market_total"] = None
            pred["over_prob"] = None
            pred["under_prob"] = None

        # Moneyline edge (model prob vs implied breakeven prob from odds)
        pred["model_edge_ml"] = _calc_edge(
            pred["home_ml_prob"], line.get("home_ml_odds")
        )

        # Spread edge
        pred["model_edge_spread"] = _calc_edge(
            pred.get("home_spread_prob"), line.get("home_spread_odds")
        )

        enriched.append(pred)

    return enriched


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability (no-vig)."""
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def _calc_edge(model_prob: float | None, odds: float | None) -> float | None:
    """Model probability minus breakeven implied probability."""
    if model_prob is None or odds is None:
        return None
    implied = _american_to_implied(odds)
    return round(model_prob - implied, 4)

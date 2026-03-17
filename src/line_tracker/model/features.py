"""Feature engineering for NCAAB point-margin prediction.

Takes per-team efficiency ratings (from ``data.py``) and builds game-level
feature vectors suitable for regression models predicting point margin.
"""

from __future__ import annotations

import pandas as pd

from line_tracker.core.logging import get_logger
from line_tracker.model.data import fetch_game_results, fetch_team_ratings

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature ordering — must be identical at train and inference time
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "adj_em_diff",
    "adj_oe_diff",
    "adj_de_diff",
    "tempo_avg",
    "tempo_diff",
    "tempo_mismatch",
    "efg_edge",
    "to_edge",
    "or_edge",
    "ftr_edge",
    "home_court",
    "tournament_flag",
    "sos_diff",
]


# ---------------------------------------------------------------------------
# Per-game feature builder
# ---------------------------------------------------------------------------


def build_game_features(
    home_team: dict,
    away_team: dict,
    *,
    neutral_site: bool = False,
    tournament: bool = False,
) -> dict:
    """Build a feature vector for a single game.

    Parameters
    ----------
    home_team, away_team:
        Dicts of team-level ratings as returned by
        ``fetch_team_ratings().to_dict("records")``.  Expected keys:
        adj_em, adj_oe, adj_de, adj_t, efg_o, efg_d, to_o, to_d,
        or_pct, ftr_o, ftr_d, sos.
    neutral_site:
        True for neutral-court games.
    tournament:
        True for NCAA tournament games (played at neutral sites with
        a reduced home-court adjustment).
    """
    # Core efficiency gap
    adj_em_diff = home_team["adj_em"] - away_team["adj_em"]
    adj_oe_diff = home_team["adj_oe"] - away_team["adj_oe"]
    adj_de_diff = home_team["adj_de"] - away_team["adj_de"]

    # Tempo
    tempo_avg = (home_team["adj_t"] + away_team["adj_t"]) / 2.0
    tempo_diff = home_team["adj_t"] - away_team["adj_t"]
    tempo_mismatch = abs(tempo_diff)

    # Four Factors matchups
    efg_edge = home_team["efg_o"] - away_team["efg_d"]
    to_edge = away_team["to_o"] - home_team["to_o"]
    or_edge = home_team["or_pct"] - away_team["or_pct"]
    ftr_edge = home_team["ftr_o"] - away_team["ftr_d"]

    # Context
    if neutral_site:
        # Tournament neutral sites still carry a slight seed-based edge (~1.5)
        home_court = 1.5 if tournament else 0.0
    else:
        home_court = 3.5

    tournament_flag = 1.0 if tournament else 0.0
    sos_diff = home_team["sos"] - away_team["sos"]

    return {
        "adj_em_diff": adj_em_diff,
        "adj_oe_diff": adj_oe_diff,
        "adj_de_diff": adj_de_diff,
        "tempo_avg": tempo_avg,
        "tempo_diff": tempo_diff,
        "tempo_mismatch": tempo_mismatch,
        "efg_edge": efg_edge,
        "to_edge": to_edge,
        "or_edge": or_edge,
        "ftr_edge": ftr_edge,
        "home_court": home_court,
        "tournament_flag": tournament_flag,
        "sos_diff": sos_diff,
    }


# ---------------------------------------------------------------------------
# Training dataset builder
# ---------------------------------------------------------------------------


def build_training_dataset(
    seasons: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix *X* and target vector *y* for given seasons.

    For each season, game results are joined with team ratings to produce
    per-game feature rows.  The target is the actual home-team point margin.

    .. note::

       **v1 approximation**: Team ratings are season-level aggregates, not
       point-in-time snapshots.  Ideally we would use ratings as of the day
       *before* each game to avoid look-ahead bias.  This is acceptable for
       an initial model but should be improved in a future version with
       rolling / pre-game ratings.

    Returns
    -------
    X : DataFrame
        Feature matrix with columns ordered by ``FEATURE_COLUMNS``.
    y : Series
        Actual home-team point margin for each game.
    """
    all_rows: list[dict] = []
    all_margins: list[float] = []

    for season in seasons:
        log.info("Building features for season %d", season)
        ratings_df = fetch_team_ratings(season)
        games_df = fetch_game_results(season)

        # Index ratings by team name for O(1) lookup
        ratings = {
            row["team"]: row
            for row in ratings_df.to_dict("records")
        }

        for _, game in games_df.iterrows():
            home = ratings.get(game["home_team"])
            away = ratings.get(game["away_team"])
            if home is None or away is None:
                # Team not found in ratings (e.g. non-D1 opponent) — skip
                continue

            is_neutral = bool(game["neutral_site"])
            features = build_game_features(
                home,
                away,
                neutral_site=is_neutral,
                # We don't have a reliable tournament flag in game_results
                # for v1; default to False.
                tournament=False,
            )
            all_rows.append(features)
            all_margins.append(float(game["margin"]))

        log.info(
            "Season %d: %d games with ratings",
            season,
            len(all_rows) - len(all_margins) + len(all_margins),
        )

    X = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS)  # noqa: N806
    y = pd.Series(all_margins, name="margin")
    log.info("Training set: %d games, %d features", len(X), len(FEATURE_COLUMNS))
    return X, y

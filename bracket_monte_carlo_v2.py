import numpy as np
from collections import Counter

from line_tracker.model.train import load_model
from line_tracker.model import data, features

# -------------------------
# SETTINGS
# -------------------------
N_SIMS = 5000
BASE_SIGMA = 10.5

# -------------------------
# LOAD MODEL + RATINGS
# -------------------------
model, scaler, _ = load_model()

ratings_df = data.fetch_team_ratings()
ratings = ratings_df.set_index("team").to_dict("index")

# -------------------------
# DEFINE BRACKET (EDIT THIS LATER)
# -------------------------
round1_games = [
    ("Duke", "Vermont"),
    ("Houston", "Prairie View A&M"),
    ("UConn", "Yale"),
    ("Arizona", "Colgate"),
]

# -------------------------
# PREDICT MARGIN
# -------------------------
def predict_margin(home, away):
    home_stats = ratings[home]
    away_stats = ratings[away]

    feat = features.build_game_features(
        home_stats,
        away_stats,
        neutral_site=True,
        tournament=True
    )

    X = np.array([list(feat.values())])
    X_scaled = scaler.transform(X)

    return model.predict(X_scaled)[0]

# -------------------------
# SIMULATE GAME (KEY UPGRADE)
# -------------------------
def simulate_game(home, away, round_num):
    margin = predict_margin(home, away)

    # increase randomness by round
    sigma = BASE_SIGMA + (round_num * 0.75)

    sim_margin = np.random.normal(margin, sigma)

    return home if sim_margin > 0 else away

# -------------------------
# SIMULATE TOURNAMENT
# -------------------------
def simulate_tournament():
    # Round 1
    round1_winners = []
    for home, away in round1_games:
        winner = simulate_game(home, away, 1)
        round1_winners.append(winner)

    # Round 2 (pair winners)
    round2_winners = []
    for i in range(0, len(round1_winners), 2):
        team1 = round1_winners[i]
        team2 = round1_winners[i + 1]
        winner = simulate_game(team1, team2, 2)
        round2_winners.append(winner)

    # Championship (simple for now)
    champ = np.random.choice(round2_winners)
    return champ

# -------------------------
# MONTE CARLO
# -------------------------
champ_counts = Counter()

for _ in range(N_SIMS):
    champ = simulate_tournament()
    champ_counts[champ] += 1

# -------------------------
# OUTPUT
# -------------------------
print("\n🏆 Championship Odds:\n")

for team, count in champ_counts.most_common():
    prob = count / N_SIMS
    print(f"{team}: {prob:.2%}")


from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import random
import re

import pandas as pd
from scipy.stats import norm

from line_tracker.model import data, features, train
from line_tracker.model.matching import TEAM_NAME_MAP


N_SIMS = 10000
SIGMA = 10.5

REGIONS = {
    "South": [
        (1, "Duke"), (16, "Siena"),
        (8, "Ohio St."), (9, "TCU"),
        (5, "Vanderbilt"), (12, "McNeese St."),
        (4, "Nebraska"), (13, "Troy"),
        (6, "North Carolina"), (11, "VCU"),
        (3, "Michigan St."), (14, "North Dakota St."),
        (7, "Saint Mary's"), (10, "Texas A&M"),
        (2, "Houston"), (15, "Idaho"),
    ],
    "Midwest": [
        (1, "Michigan"), (16, "Howard"),
        (8, "Georgia"), (9, "Saint Louis"),
        (5, "Wisconsin"), (12, "High Point"),
        (4, "Arkansas"), (13, "Hawaii"),
        (6, "BYU"), (11, "Texas"),
        (3, "Gonzaga"), (14, "Kennesaw St."),
        (7, "Kentucky"), (10, "Santa Clara"),
        (2, "Purdue"), (15, "Queens"),
    ],
    "East": [
        (1, "Florida"), (16, "Lehigh"),
        (8, "Clemson"), (9, "Iowa"),
        (5, "St. John's"), (12, "Northern Iowa"),
        (4, "Kansas"), (13, "Cal Baptist"),
        (6, "Louisville"), (11, "South Florida"),
        (3, "Illinois"), (14, "Penn"),
        (7, "Miami FL"), (10, "Missouri"),
        (2, "Connecticut"), (15, "Furman"),
    ],
    "West": [
        (1, "Arizona"), (16, "LIU"),
        (8, "Villanova"), (9, "Utah St."),
        (5, "Texas Tech"), (12, "Akron"),
        (4, "Alabama"), (13, "Hofstra"),
        (6, "Tennessee"), (11, "SMU"),
        (3, "Virginia"), (14, "Wright St."),
        (7, "UCLA"), (10, "UCF"),
        (2, "Iowa St."), (15, "Tennessee St."),
    ],
}

FINAL_FOUR_PAIRINGS = [("South", "Midwest"), ("East", "West")]
ROUND_ORDER = ["S16", "E8", "F4", "Final", "Champ"]


def _normalize_for_cmp(name: str) -> str:
    s = name.lower().replace(".", "")
    s = re.sub(r"\bstate\b", "st", s)
    return re.sub(r"\s+", " ", s).strip()


def _build_name_index(torvik_names: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}

    for name in torvik_names:
        index[name.lower()] = name

    for name in torvik_names:
        low = name.lower()
        if " st." in low:
            index.setdefault(low.replace(" st.", " state"), name)
        elif " state" in low:
            index.setdefault(low.replace(" state", " st."), name)

    torvik_set = set(torvik_names)
    torvik_lower_map = {n.lower(): n for n in torvik_names}
    for espn_lower, torvik in TEAM_NAME_MAP.items():
        if torvik in torvik_set or torvik.lower() in torvik_lower_map:
            index[espn_lower] = torvik

    return index


def _resolve_team(name: str, name_index: dict[str, str]) -> str | None:
    low = name.lower().strip()

    if low in name_index:
        return name_index[low]

    low_cmp = _normalize_for_cmp(low)
    for key, torvik in name_index.items():
        if _normalize_for_cmp(key) == low_cmp:
            return torvik

    best = None
    best_len = 0
    for key, torvik in name_index.items():
        key_cmp = _normalize_for_cmp(key)
        if low_cmp.startswith(key_cmp) and len(key_cmp) > best_len:
            if len(low_cmp) == len(key_cmp) or low_cmp[len(key_cmp)] == " ":
                best = torvik
                best_len = len(key_cmp)
    return best


@dataclass(frozen=True)
class Team:
    seed: int
    display: str
    canonical: str
    region: str


class TrainedBracketSimulator:
    def __init__(self, sigma: float = SIGMA):
        self.sigma = sigma
        self.model, self.scaler, self.metadata = train.load_model(None)
        self.ratings_df = data.fetch_team_ratings()
        self.ratings_idx = self.ratings_df.set_index("team")
        self.name_index = _build_name_index(list(self.ratings_idx.index))
        self.teams = self._build_teams()
        self.seed_map = {t.display: t.seed for t in self.teams.values()}
        self.region_map = {t.display: t.region for t in self.teams.values()}
        self.matchup_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._precache_matchups()

    def _build_teams(self) -> dict[str, Team]:
        teams: dict[str, Team] = {}
        for region, slot_list in REGIONS.items():
            for seed, display in slot_list:
                canonical = _resolve_team(display, self.name_index)
                if canonical is None or canonical not in self.ratings_idx.index:
                    raise ValueError(f"Could not resolve team '{display}' to current ratings data.")
                teams[display] = Team(seed=seed, display=display, canonical=canonical, region=region)
        return teams

    def _predict_margin_one_way(self, team_a: str, team_b: str) -> float:
        a = self.teams[team_a]
        b = self.teams[team_b]
        a_stats = self.ratings_idx.loc[a.canonical].to_dict()
        b_stats = self.ratings_idx.loc[b.canonical].to_dict()

        feat = features.build_game_features(
            a_stats,
            b_stats,
            neutral_site=True,
            tournament=True,
        )
        X = pd.DataFrame([feat])[features.FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])

    def predict_neutral_matchup(self, team_a: str, team_b: str) -> tuple[float, float]:
        key = tuple(sorted((team_a, team_b)))
        cached = self.matchup_cache.get(key)
        if cached is not None:
            if key[0] == team_a:
                return cached
            return (-cached[0], 1.0 - cached[1])

        ab = self._predict_margin_one_way(team_a, team_b)
        ba = self._predict_margin_one_way(team_b, team_a)

        neutral_margin = (ab - ba) / 2.0
        team_a_win_prob = float(norm.cdf(neutral_margin / self.sigma))

        if key[0] == team_a:
            self.matchup_cache[key] = (neutral_margin, team_a_win_prob)
            return neutral_margin, team_a_win_prob

        self.matchup_cache[key] = (-neutral_margin, 1.0 - team_a_win_prob)
        return neutral_margin, team_a_win_prob

    def _precache_matchups(self) -> None:
        team_names = list(self.teams.keys())
        for team_a, team_b in combinations(team_names, 2):
            self.predict_neutral_matchup(team_a, team_b)

    def simulate_game(self, team_a: str, team_b: str, rng: random.Random) -> str:
        _, p_a = self.predict_neutral_matchup(team_a, team_b)
        return team_a if rng.random() < p_a else team_b

    def deterministic_game(self, team_a: str, team_b: str) -> tuple[str, float, float]:
        margin, p_a = self.predict_neutral_matchup(team_a, team_b)
        winner = team_a if p_a >= 0.5 else team_b
        winner_prob = p_a if winner == team_a else (1.0 - p_a)
        winner_margin = margin if winner == team_a else -margin
        return winner, winner_prob, winner_margin

    def simulate_region(self, region_name: str, rng: random.Random, counts: dict[str, Counter]) -> str:
        teams = [team for _, team in REGIONS[region_name]]

        r64_pairs = [(teams[i], teams[i + 1]) for i in range(0, 16, 2)]
        r32 = []
        for a, b in r64_pairs:
            r32.append(self.simulate_game(a, b, rng))

        s16 = []
        for i in range(0, 8, 2):
            w = self.simulate_game(r32[i], r32[i + 1], rng)
            s16.append(w)
            counts["sweet16"][w] += 1

        e8 = []
        for i in range(0, 4, 2):
            w = self.simulate_game(s16[i], s16[i + 1], rng)
            e8.append(w)
            counts["elite8"][w] += 1

        champ = self.simulate_game(e8[0], e8[1], rng)
        counts["final4"][champ] += 1
        counts["region_titles"][champ] += 1
        return champ


def build_team_round_probs(counts: dict[str, Counter], n_sims: int) -> dict[str, dict[str, float]]:
    round_map = {
        "S16": counts["sweet16"],
        "E8": counts["elite8"],
        "F4": counts["final4"],
        "Final": counts["finals"],
        "Champ": counts["championship"],
    }

    teams = set()
    for counter in round_map.values():
        teams.update(counter.keys())

    out: dict[str, dict[str, float]] = {}
    for team in teams:
        out[team] = {}
        for round_name, counter in round_map.items():
            out[team][round_name] = counter[team] / n_sims
    return out


def run_trained_bracket_simulation(n_sims: int = N_SIMS, seed: int = 20260318) -> dict:
    sim = TrainedBracketSimulator(sigma=SIGMA)
    rng = random.Random(seed)

    counts = {
        "championship": Counter(),
        "finals": Counter(),
        "final4": Counter(),
        "elite8": Counter(),
        "sweet16": Counter(),
        "region_titles": Counter(),
        "first_round_upsets": Counter(),
    }

    region_winner_counts = {region: Counter() for region in REGIONS}

    for _ in range(n_sims):
        regional_champs = {}
        for region_name in REGIONS:
            champ = sim.simulate_region(region_name, rng, counts)
            regional_champs[region_name] = champ
            region_winner_counts[region_name][champ] += 1

        ff_winners = []
        for left, right in FINAL_FOUR_PAIRINGS:
            semifinal_winner = sim.simulate_game(regional_champs[left], regional_champs[right], rng)
            ff_winners.append(semifinal_winner)
            counts["finals"][semifinal_winner] += 1

        national_champ = sim.simulate_game(ff_winners[0], ff_winners[1], rng)
        counts["championship"][national_champ] += 1

        for region_name, slot_list in REGIONS.items():
            teams = [team for _, team in slot_list]
            for i in range(0, 16, 2):
                a, b = teams[i], teams[i + 1]
                winner = sim.simulate_game(a, b, rng)
                seed_a = sim.seed_map[a]
                seed_b = sim.seed_map[b]
                higher = a if seed_a < seed_b else b
                lower = b if higher == a else a
                if winner == lower:
                    counts["first_round_upsets"][(lower, higher)] += 1

    team_round_probs = build_team_round_probs(counts, n_sims)

    return {
        "n_sims": n_sims,
        "seed": seed,
        "sim": sim,
        "counts": counts,
        "region_winner_counts": region_winner_counts,
        "team_round_probs": team_round_probs,
    }

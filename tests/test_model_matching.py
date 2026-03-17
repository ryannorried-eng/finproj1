"""Tests for Odds API → Torvik team name matching."""

from __future__ import annotations

import pytest

from line_tracker.model.matching import (
    TEAM_NAME_MAP,
    _fuzzy_match_team,
    _normalize_for_cmp,
    _substring_match,
    match_all_events,
    match_event_to_prediction,
    normalize_team_name,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred(home: str, away: str) -> dict:
    """Create a minimal prediction dict."""
    return {
        "home_team": home,
        "away_team": away,
        "predicted_margin": 3.5,
    }


# ---------------------------------------------------------------------------
# normalize_team_name
# ---------------------------------------------------------------------------

class TestNormalizeTeamName:
    def test_strips_mascot_and_lowercases(self):
        assert normalize_team_name("Arkansas Razorbacks") == "arkansas"

    def test_strips_multi_word_mascot(self):
        assert normalize_team_name("Duke Blue Devils") == "duke"

    def test_strips_whitespace(self):
        assert normalize_team_name("  Duke  Blue Devils ") == "duke"

    def test_preserves_parenthetical(self):
        assert normalize_team_name("Miami (OH) RedHawks") == "miami (oh)"

    def test_multi_word_school_name(self):
        assert normalize_team_name("Texas Tech Red Raiders") == "texas tech"

    def test_no_mascot(self):
        assert normalize_team_name("Duke") == "duke"

    def test_ampersand_name(self):
        assert normalize_team_name("Prairie View A&M Panthers") == "prairie view a&m"

    def test_st_prefix(self):
        assert normalize_team_name("St. John's Red Storm") == "st. john's"

    def test_horned_frogs(self):
        assert normalize_team_name("TCU Horned Frogs") == "tcu"

    def test_fighting_illini(self):
        assert normalize_team_name("Illinois Fighting Illini") == "illinois"

    def test_mountain_hawks(self):
        assert normalize_team_name("Lehigh Mountain Hawks") == "lehigh"


# ---------------------------------------------------------------------------
# TEAM_NAME_MAP coverage
# ---------------------------------------------------------------------------

class TestTeamNameMap:
    """Verify the manual map covers all 68 tournament teams."""

    TOURNAMENT_ODDS_NAMES = [
        # 1-seeds
        "Duke Blue Devils",
        "Houston Cougars",
        "Florida Gators",
        "Auburn Tigers",
        # 2-seeds
        "Alabama Crimson Tide",
        "Tennessee Volunteers",
        "Michigan State Spartans",
        "St. John's Red Storm",
        # 3-seeds
        "Iowa State Cyclones",
        "Texas Tech Red Raiders",
        "Wisconsin Badgers",
        "Connecticut Huskies",
        # 4-seeds
        "Arizona Wildcats",
        "Purdue Boilermakers",
        "Maryland Terrapins",
        "Gonzaga Bulldogs",
        # 5-seeds
        "Michigan Wolverines",
        "Clemson Tigers",
        "Louisville Cardinals",
        "Texas Longhorns",
        # 6-seeds
        "Illinois Fighting Illini",
        "BYU Cougars",
        "Missouri Tigers",
        "Oklahoma Sooners",
        # 7-seeds
        "Marquette Golden Eagles",
        "Kansas Jayhawks",
        "UCLA Bruins",
        "Mississippi State Bulldogs",
        # 8-seeds
        "Ohio State Buckeyes",
        "Georgia Bulldogs",
        "South Carolina Gamecocks",
        "Nebraska Cornhuskers",
        # 9-seeds
        "Arkansas Razorbacks",
        "Creighton Bluejays",
        "Baylor Bears",
        "Louisiana Tech Bulldogs",
        # 10-seeds
        "Vanderbilt Commodores",
        "New Mexico Lobos",
        "North Carolina Tar Heels",
        "Utah State Aggies",
        # 11-seeds
        "SMU Mustangs",
        "Virginia Cavaliers",
        "TCU Horned Frogs",
        "Xavier Musketeers",
        "San Diego State Aztecs",
        "Drake Bulldogs",
        "NC State Wolfpack",
        # 12-seeds
        "UCF Knights",
        "Liberty Flames",
        "McNeese Cowboys",
        "Colorado State Rams",
        "VCU Rams",
        # 13-seeds
        "Yale Bulldogs",
        "High Point Panthers",
        "Vermont Catamounts",
        "Troy Trojans",
        # 14-seeds
        "Lipscomb Bisons",
        "South Florida Bulls",
        "Lehigh Mountain Hawks",
        "Siena Saints",
        # 15-seeds
        "Prairie View A&M Panthers",
        "Omaha Mavericks",
        "Southeast Missouri State Redhawks",
        "Miami (OH) Redhawks",
        # 16-seeds
        "Norfolk State Spartans",
        "Howard Bison",
        "UMBC Retrievers",
        "Alabama State Hornets",
    ]

    def test_all_tournament_teams_in_map(self):
        """Every expected Odds API name should resolve via the map."""
        missing = []
        for name in self.TOURNAMENT_ODDS_NAMES:
            if name.lower() not in TEAM_NAME_MAP:
                missing.append(name)
        assert missing == [], f"Missing from TEAM_NAME_MAP: {missing}"

    def test_map_values_are_torvik_format(self):
        """Torvik names should not contain mascots."""
        mascot_words = {"bulldogs", "wildcats", "tigers", "bears"}
        for odds_name, torvik_name in TEAM_NAME_MAP.items():
            for w in mascot_words:
                assert w not in torvik_name.lower(), (
                    f"TEAM_NAME_MAP[{odds_name!r}] = {torvik_name!r} looks like "
                    f"it still contains a mascot word ({w!r})"
                )


# ---------------------------------------------------------------------------
# match_event_to_prediction — exact map matches
# ---------------------------------------------------------------------------

class TestMatchEventExact:
    PREDICTIONS = [
        _pred("Duke", "Arkansas"),
        _pred("Houston", "Tennessee"),
        _pred("Connecticut", "Purdue"),
    ]

    def test_exact_map_match(self):
        event = {"home_team": "Duke Blue Devils", "away_team": "Arkansas Razorbacks"}
        result = match_event_to_prediction(event, self.PREDICTIONS)
        assert result is not None
        assert result["home_team"] == "Duke"
        assert result["away_team"] == "Arkansas"

    def test_exact_map_match_uconn(self):
        event = {"home_team": "Connecticut Huskies", "away_team": "Purdue Boilermakers"}
        result = match_event_to_prediction(event, self.PREDICTIONS)
        assert result is not None
        assert result["home_team"] == "Connecticut"

    def test_no_match_returns_none(self):
        event = {"home_team": "Fake School Lions", "away_team": "Other School Bears"}
        result = match_event_to_prediction(event, self.PREDICTIONS)
        assert result is None


# ---------------------------------------------------------------------------
# match_event_to_prediction — normalized / fuzzy fallback
# ---------------------------------------------------------------------------

class TestMatchEventNormalized:
    PREDICTIONS = [
        _pred("Wisconsin", "Marquette"),
        _pred("Miami OH", "Lehigh"),
    ]

    def test_normalized_fallback(self):
        """Teams not in the map should still match via mascot stripping."""
        event = {
            "home_team": "Wisconsin Badgers",
            "away_team": "Marquette Golden Eagles",
        }
        result = match_event_to_prediction(event, self.PREDICTIONS)
        assert result is not None
        assert result["home_team"] == "Wisconsin"

    def test_mismatch_does_not_return_wrong_team(self):
        """Don't return a match when only one team matches."""
        event = {"home_team": "Wisconsin Badgers", "away_team": "Lehigh Mountain Hawks"}
        result = match_event_to_prediction(event, self.PREDICTIONS)
        # There's no Wisconsin vs Lehigh prediction → should be None
        assert result is None


# ---------------------------------------------------------------------------
# match_all_events
# ---------------------------------------------------------------------------

class TestMatchAllEvents:
    def test_batch_matching(self):
        predictions = [
            _pred("Duke", "Arkansas"),
            _pred("Houston", "Tennessee"),
        ]
        events = [
            {
                "id": "evt_1",
                "home_team": "Duke Blue Devils",
                "away_team": "Arkansas Razorbacks",
            },
            {
                "id": "evt_2",
                "home_team": "Houston Cougars",
                "away_team": "Tennessee Volunteers",
            },
            {
                "id": "evt_3",
                "home_team": "Random Team",
                "away_team": "Unknown School",
            },
        ]
        result = match_all_events(events, predictions)
        assert "evt_1" in result
        assert "evt_2" in result
        assert "evt_3" not in result
        assert result["evt_1"]["home_team"] == "Duke"

    def test_empty_events(self):
        result = match_all_events([], [_pred("Duke", "Arkansas")])
        assert result == {}

    def test_empty_predictions(self):
        events = [
            {
                "id": "e1",
                "home_team": "Duke Blue Devils",
                "away_team": "Arkansas Razorbacks",
            },
        ]
        result = match_all_events(events, [])
        assert result == {}


# ---------------------------------------------------------------------------
# Real NCAA tournament matchup scenarios
# ---------------------------------------------------------------------------

class TestRealTournamentMatchups:
    """Test with realistic first-round tournament matchups."""

    PREDICTIONS = [
        _pred("Duke", "Howard"),
        _pred("Florida", "Norfolk St."),
        _pred("Houston", "UMBC"),
        _pred("Auburn", "Alabama St."),
        _pred("Alabama", "McNeese St."),
        _pred("Tennessee", "Miami OH"),
        _pred("Michigan St.", "Lehigh"),
        _pred("St. John's", "Omaha"),  # Nebraska Omaha in Torvik
        _pred("Iowa St.", "Lipscomb"),
        _pred("Texas Tech", "Prairie View A&M"),
        _pred("Wisconsin", "High Point"),
        _pred("Connecticut", "Siena"),
        _pred("Arizona", "Troy"),
        _pred("Purdue", "South Fla."),
        _pred("Maryland", "Vermont"),
        _pred("Gonzaga", "Yale"),
        _pred("Michigan", "UCF"),
        _pred("Clemson", "Liberty"),
        _pred("Louisville", "McNeese St."),
        _pred("Texas", "Colorado St."),
        _pred("SMU", "VCU"),
        _pred("Virginia", "Drake"),
        _pred("TCU", "Xavier"),
        _pred("N.C. State", "San Diego St."),
    ]

    @pytest.mark.parametrize(
        "odds_home, odds_away, expected_home, expected_away",
        [
            ("Duke Blue Devils", "Howard Bison", "Duke", "Howard"),
            ("Florida Gators", "Norfolk State Spartans", "Florida", "Norfolk St."),
            ("Houston Cougars", "UMBC Retrievers", "Houston", "UMBC"),
            ("Auburn Tigers", "Alabama State Hornets", "Auburn", "Alabama St."),
            ("Alabama Crimson Tide", "McNeese Cowboys", "Alabama", "McNeese St."),
            ("Tennessee Volunteers", "Miami (OH) Redhawks", "Tennessee", "Miami OH"),
            ("Michigan State Spartans", "Lehigh Mountain Hawks",
             "Michigan St.", "Lehigh"),
            ("Iowa State Cyclones", "Lipscomb Bisons",
             "Iowa St.", "Lipscomb"),
            ("Texas Tech Red Raiders", "Prairie View A&M Panthers",
             "Texas Tech", "Prairie View A&M"),
            ("Wisconsin Badgers", "High Point Panthers", "Wisconsin", "High Point"),
            ("Connecticut Huskies", "Siena Saints", "Connecticut", "Siena"),
            ("Arizona Wildcats", "Troy Trojans", "Arizona", "Troy"),
            ("Purdue Boilermakers", "South Florida Bulls", "Purdue", "South Fla."),
            ("Maryland Terrapins", "Vermont Catamounts", "Maryland", "Vermont"),
            ("Gonzaga Bulldogs", "Yale Bulldogs", "Gonzaga", "Yale"),
            ("Michigan Wolverines", "UCF Knights", "Michigan", "UCF"),
            ("Clemson Tigers", "Liberty Flames", "Clemson", "Liberty"),
            ("SMU Mustangs", "VCU Rams", "SMU", "VCU"),
            ("Virginia Cavaliers", "Drake Bulldogs", "Virginia", "Drake"),
            ("TCU Horned Frogs", "Xavier Musketeers", "TCU", "Xavier"),
            ("NC State Wolfpack", "San Diego State Aztecs",
             "N.C. State", "San Diego St."),
        ],
    )
    def test_tournament_matchup(
        self, odds_home, odds_away, expected_home, expected_away,
    ):
        event = {"home_team": odds_home, "away_team": odds_away}
        result = match_event_to_prediction(event, self.PREDICTIONS)
        assert result is not None, f"No match for {odds_away} @ {odds_home}"
        assert result["home_team"] == expected_home
        assert result["away_team"] == expected_away

    def test_omaha_matches_nebraska_omaha(self):
        """Omaha Mavericks should match the Torvik name 'Nebraska Omaha'."""
        # This requires the map override — Omaha → Nebraska Omaha
        preds = [_pred("St. John's", "Nebraska Omaha")]
        event = {"home_team": "St. John's Red Storm", "away_team": "Omaha Mavericks"}
        result = match_event_to_prediction(event, preds)
        assert result is not None
        assert result["away_team"] == "Nebraska Omaha"

    def test_se_missouri_state(self):
        """Southeast Missouri State Redhawks → Southeast Missouri St."""
        preds = [_pred("Tennessee", "Southeast Missouri St.")]
        event = {
            "home_team": "Tennessee Volunteers",
            "away_team": "Southeast Missouri State Redhawks",
        }
        result = match_event_to_prediction(event, preds)
        assert result is not None
        assert result["away_team"] == "Southeast Missouri St."


# ---------------------------------------------------------------------------
# State vs St. disambiguation (regression tests for name mismatch bugs)
# ---------------------------------------------------------------------------

class TestStateStDotDisambiguation:
    """Ensure 'Iowa State' never matches 'Iowa', etc."""

    def test_normalize_for_cmp_state_to_st(self):
        assert _normalize_for_cmp("Iowa State") == "iowa st"
        assert _normalize_for_cmp("Iowa St.") == "iowa st"
        assert _normalize_for_cmp("Iowa") == "iowa"

    def test_fuzzy_iowa_state_vs_iowa(self):
        """Iowa State Cyclones must match Iowa St., not Iowa."""
        teams = {"Iowa", "Iowa St."}
        assert _fuzzy_match_team("Iowa State Cyclones", teams) == "Iowa St."
        assert _fuzzy_match_team("Iowa State", teams) == "Iowa St."

    def test_fuzzy_iowa_alone(self):
        """Plain 'Iowa' should match 'Iowa', not 'Iowa St.'."""
        teams = {"Iowa", "Iowa St."}
        assert _fuzzy_match_team("Iowa Hawkeyes", teams) == "Iowa"
        assert _fuzzy_match_team("Iowa", teams) == "Iowa"

    def test_fuzzy_illinois_vs_illinois_state(self):
        teams = {"Illinois", "Illinois St."}
        assert _fuzzy_match_team("Illinois Fighting Illini", teams) == "Illinois"
        assert _fuzzy_match_team("Illinois State Redbirds", teams) == "Illinois St."
        assert _fuzzy_match_team("Illinois St", teams) == "Illinois St."

    def test_fuzzy_ohio_state_vs_ohio(self):
        teams = {"Ohio", "Ohio St."}
        assert _fuzzy_match_team("Ohio State Buckeyes", teams) == "Ohio St."
        assert _fuzzy_match_team("Ohio Bobcats", teams) == "Ohio"

    def test_fuzzy_utah_state_vs_utah(self):
        teams = {"Utah", "Utah St."}
        assert _fuzzy_match_team("Utah State Aggies", teams) == "Utah St."
        assert _fuzzy_match_team("Utah Utes", teams) == "Utah"

    def test_substring_iowa_state_vs_iowa(self):
        """_substring_match should also pick the right team."""
        teams = {"Iowa", "Iowa St."}
        assert _substring_match("Iowa State Cyclones", teams) == "Iowa St."

    def test_substring_illinois_vs_illinois_state(self):
        teams = {"Illinois", "Illinois St."}
        assert _substring_match("Illinois Fighting Illini", teams) == "Illinois"

    def test_full_event_iowa_state_not_iowa(self):
        """End-to-end: Iowa State must not match Iowa in predictions."""
        preds = [
            _pred("Iowa St.", "Lipscomb"),
            _pred("Iowa", "Vermont"),
        ]
        event = {
            "home_team": "Iowa State Cyclones",
            "away_team": "Lipscomb Bisons",
        }
        result = match_event_to_prediction(event, preds)
        assert result is not None
        assert result["home_team"] == "Iowa St."

    def test_full_event_iowa_not_iowa_state(self):
        preds = [
            _pred("Iowa St.", "Lipscomb"),
            _pred("Iowa", "Vermont"),
        ]
        event = {
            "home_team": "Iowa Hawkeyes",
            "away_team": "Vermont Catamounts",
        }
        result = match_event_to_prediction(event, preds)
        assert result is not None
        assert result["home_team"] == "Iowa"

    def test_full_event_utah_state_at_villanova(self):
        preds = [
            _pred("Villanova", "Utah St."),
            _pred("Utah", "Yale"),
        ]
        event = {
            "home_team": "Villanova Wildcats",
            "away_team": "Utah State Aggies",
        }
        result = match_event_to_prediction(event, preds)
        assert result is not None
        assert result["away_team"] == "Utah St."

    def test_full_event_tcu_at_ohio_state(self):
        preds = [
            _pred("Ohio St.", "TCU"),
            _pred("Ohio", "Drake"),
        ]
        event = {
            "home_team": "Ohio State Buckeyes",
            "away_team": "TCU Horned Frogs",
        }
        result = match_event_to_prediction(event, preds)
        assert result is not None
        assert result["home_team"] == "Ohio St."
        assert result["away_team"] == "TCU"

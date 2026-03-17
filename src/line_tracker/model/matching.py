"""Match Odds API event names to Barttorvik/model prediction team names.

The Odds API uses full display names (e.g. "Arkansas Razorbacks") while
Barttorvik uses short canonical names (e.g. "Arkansas").  This module
provides a three-tier matching strategy:

1. **Exact lookup** via ``TEAM_NAME_MAP`` – manually curated overrides.
2. **Normalized matching** – strip mascot suffixes, lowercase, collapse
   whitespace, then compare.
3. **Substring fallback** – check if the Torvik name appears as a prefix
   or substring of the Odds API name.
"""

from __future__ import annotations

import re

from line_tracker.core.logging import get_logger

log = get_logger(__name__)


def _normalize_for_cmp(name: str) -> str:
    """Normalize for comparison: lowercase, strip periods, ``State`` → ``St``.

    This ensures that Torvik names like ``"Iowa St."`` and normalised Odds API
    names like ``"iowa state"`` compare as equal.
    """
    s = name.lower().replace(".", "")
    s = re.sub(r"\bstate\b", "st", s)
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------------------
# 1.  Manual override map: Odds API display name  →  Torvik canonical name
#
#     Keys are *lowercased* Odds API names.  This is the primary matching
#     method and should cover every team we care about.
# ---------------------------------------------------------------------------

TEAM_NAME_MAP: dict[str, str] = {
    # ── 2026 NCAA Tournament field (68 teams) ──────────────────────────
    # 1-seeds
    "duke blue devils": "Duke",
    "houston cougars": "Houston",
    "florida gators": "Florida",
    "auburn tigers": "Auburn",
    # 2-seeds
    "alabama crimson tide": "Alabama",
    "tennessee volunteers": "Tennessee",
    "michigan state spartans": "Michigan St.",
    "st. john's red storm": "St. John's",
    "st john's red storm": "St. John's",
    # 3-seeds
    "iowa state cyclones": "Iowa St.",
    "texas tech red raiders": "Texas Tech",
    "wisconsin badgers": "Wisconsin",
    "connecticut huskies": "Connecticut",
    "uconn huskies": "Connecticut",
    # 4-seeds
    "arizona wildcats": "Arizona",
    "purdue boilermakers": "Purdue",
    "maryland terrapins": "Maryland",
    "gonzaga bulldogs": "Gonzaga",
    # 5-seeds
    "michigan wolverines": "Michigan",
    "clemson tigers": "Clemson",
    "louisville cardinals": "Louisville",
    "texas longhorns": "Texas",
    # 6-seeds
    "illinois fighting illini": "Illinois",
    "byu cougars": "BYU",
    "missouri tigers": "Missouri",
    "oklahoma sooners": "Oklahoma",
    # 7-seeds
    "marquette golden eagles": "Marquette",
    "kansas jayhawks": "Kansas",
    "ucla bruins": "UCLA",
    "mississippi state bulldogs": "Mississippi St.",
    # 8-seeds
    "ohio state buckeyes": "Ohio St.",
    "georgia bulldogs": "Georgia",
    "south carolina gamecocks": "South Carolina",
    "nebraska cornhuskers": "Nebraska",
    # 9-seeds
    "arkansas razorbacks": "Arkansas",
    "creighton bluejays": "Creighton",
    "baylor bears": "Baylor",
    "louisiana tech bulldogs": "Louisiana Tech",
    # 10-seeds
    "vanderbilt commodores": "Vanderbilt",
    "new mexico lobos": "New Mexico",
    "north carolina tar heels": "North Carolina",
    "utah state aggies": "Utah St.",
    # 11-seeds
    "smu mustangs": "SMU",
    "virginia cavaliers": "Virginia",
    "tcu horned frogs": "TCU",
    "xavier musketeers": "Xavier",
    "san diego state aztecs": "San Diego St.",
    "drake bulldogs": "Drake",
    "nc state wolfpack": "N.C. State",
    "north carolina state wolfpack": "N.C. State",
    # 12-seeds
    "ucf knights": "UCF",
    "liberty flames": "Liberty",
    "mcneese cowboys": "McNeese St.",
    "mcneese state cowboys": "McNeese St.",
    "colorado state rams": "Colorado St.",
    "vcu rams": "VCU",
    # 13-seeds
    "yale bulldogs": "Yale",
    "high point panthers": "High Point",
    "vermont catamounts": "Vermont",
    "troy trojans": "Troy",
    # 14-seeds
    "lipscomb bisons": "Lipscomb",
    "south florida bulls": "South Fla.",
    "usf bulls": "South Fla.",
    "lehigh mountain hawks": "Lehigh",
    "siena saints": "Siena",
    # 15-seeds
    "prairie view a&m panthers": "Prairie View A&M",
    "omaha mavericks": "Nebraska Omaha",
    "southeast missouri state redhawks": "Southeast Missouri St.",
    "miami (oh) redhawks": "Miami OH",
    "miami redhawks": "Miami OH",
    "miami ohio redhawks": "Miami OH",
    # 16-seeds
    "norfolk state spartans": "Norfolk St.",
    "howard bison": "Howard",
    "umbc retrievers": "UMBC",
    "alabama state hornets": "Alabama St.",
    # ── Additional top-100 / bubble teams ──────────────────────────────
    "kentucky wildcats": "Kentucky",
    "indiana hoosiers": "Indiana",
    "oregon ducks": "Oregon",
    "pitt panthers": "Pittsburgh",
    "pittsburgh panthers": "Pittsburgh",
    "memphis tigers": "Memphis",
    "cincinnati bearcats": "Cincinnati",
    "dayton flyers": "Dayton",
    "colorado buffaloes": "Colorado",
    "wake forest demon deacons": "Wake Forest",
    "florida state seminoles": "Florida St.",
    "virginia tech hokies": "Virginia Tech",
    "providence friars": "Providence",
    "villanova wildcats": "Villanova",
    "miami hurricanes": "Miami FL",
    "iowa hawkeyes": "Iowa",
    "penn state nittany lions": "Penn St.",
    "kansas state wildcats": "Kansas St.",
    "west virginia mountaineers": "West Virginia",
    "georgia tech yellow jackets": "Georgia Tech",
    "ole miss rebels": "Ole Miss",
    "mississippi rebels": "Ole Miss",
    "rutgers scarlet knights": "Rutgers",
    "southern california trojans": "USC",
    "usc trojans": "USC",
    "stanford cardinal": "Stanford",
    "washington huskies": "Washington",
    "northwestern wildcats": "Northwestern",
    "minnesota golden gophers": "Minnesota",
    "syracuse orange": "Syracuse",
    "texas a&m aggies": "Texas A&M",
    "lsu tigers": "LSU",
    "south dakota state jackrabbits": "South Dakota St.",
    "wichita state shockers": "Wichita St.",
    "saint mary's gaels": "Saint Mary's",
    "st. mary's gaels": "Saint Mary's",
    "nevada wolf pack": "Nevada",
    "new mexico state aggies": "New Mexico St.",
    "unlv rebels": "UNLV",
    "utep miners": "UTEP",
    "utsa roadrunners": "UTSA",
    "unc asheville bulldogs": "UNC Asheville",
    "unc greensboro spartans": "UNC Greensboro",
    "unc wilmington seahawks": "UNC Wilmington",
}

# ---------------------------------------------------------------------------
# 2.  Common mascot suffixes used by ESPN / Odds API
# ---------------------------------------------------------------------------

_MASCOT_SUFFIXES: list[str] = [
    "blue devils",
    "tar heels",
    "crimson tide",
    "fighting illini",
    "golden eagles",
    "mountain hawks",
    "scarlet knights",
    "red raiders",
    "red storm",
    "yellow jackets",
    "demon deacons",
    "horned frogs",
    "nittany lions",
    "wolf pack",
    "razorbacks",
    "boilermakers",
    "wildcats",
    "bulldogs",
    "huskies",
    "cougars",
    "mustangs",
    "cavaliers",
    "volunteers",
    "longhorns",
    "sooners",
    "cyclones",
    "jayhawks",
    "seminoles",
    "cardinals",
    "gamecocks",
    "cornhuskers",
    "wolverines",
    "spartans",
    "tigers",
    "bears",
    "cowboys",
    "musketeers",
    "aztecs",
    "bruins",
    "gators",
    "gaels",
    "hoosiers",
    "ducks",
    "panthers",
    "rams",
    "flames",
    "catamounts",
    "trojans",
    "saints",
    "bison",
    "retrievers",
    "hornets",
    "mavericks",
    "redhawks",
    "commodores",
    "lobos",
    "aggies",
    "flyers",
    "buffaloes",
    "friars",
    "orange",
    "cardinal",
    "hokies",
    "knights",
    "bearcats",
    "jackrabbits",
    "shockers",
    "hawks",
    "eagles",
    "owls",
    "terriers",
    "bobcats",
    "billikens",
    "terrapins",
    "bluejays",
    "foxes",
    "bears",
]

# Build a single regex that strips the longest mascot suffix.
# Sort by length descending so longer suffixes match first.
_sorted_mascots = sorted(_MASCOT_SUFFIXES, key=len, reverse=True)
_MASCOT_PATTERN = re.compile(
    r"\s+(?:" + "|".join(re.escape(s) for s in _sorted_mascots) + r")$",
    re.IGNORECASE,
)


def normalize_team_name(name: str) -> str:
    """Normalize a team name by stripping mascot, lowercasing, collapsing ws.

    >>> normalize_team_name("Arkansas Razorbacks")
    'arkansas'
    >>> normalize_team_name("  Duke  Blue Devils ")
    'duke'
    >>> normalize_team_name("Miami (OH) RedHawks")
    'miami (oh)'
    """
    text = name.strip()
    text = _MASCOT_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ---------------------------------------------------------------------------
# 3.  Event → prediction matching
# ---------------------------------------------------------------------------


def _torvik_name_for(odds_name: str) -> str | None:
    """Look up a single Odds API team name in the manual map."""
    return TEAM_NAME_MAP.get(odds_name.lower().strip())


def _find_prediction_by_teams(
    home_torvik: str,
    away_torvik: str,
    predictions: list[dict],
) -> dict | None:
    """Find the prediction that matches the given Torvik home/away names."""
    for pred in predictions:
        if pred["home_team"] == home_torvik and pred["away_team"] == away_torvik:
            return pred
    return None


def _fuzzy_match_team(
    odds_name: str,
    prediction_teams: set[str],
) -> str | None:
    """Attempt normalized / substring matching against known prediction teams.

    Returns the matched Torvik team name or None.
    """
    norm = normalize_team_name(odds_name)
    norm_cmp = _normalize_for_cmp(norm)

    # Exact match: raw normalised name or comparison-normalised name
    for t in prediction_teams:
        if norm == t.lower() or norm_cmp == _normalize_for_cmp(t):
            return t

    # Word-boundary prefix matching – longest match wins so that
    # "Iowa St." (longer) beats "Iowa" when the input is "Iowa State …".
    best: str | None = None
    best_len = 0
    for t in prediction_teams:
        t_cmp = _normalize_for_cmp(t)
        # Torvik name is a prefix of the odds name
        if norm_cmp.startswith(t_cmp) and len(t_cmp) > best_len:
            # Require word boundary after the prefix
            if len(norm_cmp) == len(t_cmp) or norm_cmp[len(t_cmp)] == " ":
                best = t
                best_len = len(t_cmp)
        # Odds name is a prefix of the Torvik name
        if t_cmp.startswith(norm_cmp) and len(norm_cmp) > best_len:
            if len(t_cmp) == len(norm_cmp) or t_cmp[len(norm_cmp)] == " ":
                best = t
                best_len = len(norm_cmp)
    return best


def match_event_to_prediction(
    odds_event: dict,
    predictions: list[dict],
) -> dict | None:
    """Match a single Odds API event to its model prediction.

    Matching tiers
    --------------
    a) Exact ``TEAM_NAME_MAP`` lookup for both home and away teams.
    b) Normalized name matching (mascot-stripped, lowered).
    c) Substring / prefix matching as a last resort.

    Parameters
    ----------
    odds_event
        Must contain ``home_team`` and ``away_team`` string fields.
    predictions
        List of prediction dicts, each with ``home_team`` and ``away_team``
        keyed by Torvik canonical names.

    Returns
    -------
    dict | None
        The matching prediction dict, or ``None`` if no match found.
    """
    odds_home = odds_event["home_team"]
    odds_away = odds_event["away_team"]

    # Collect the set of Torvik names present in predictions
    pred_teams: set[str] = set()
    for p in predictions:
        pred_teams.add(p["home_team"])
        pred_teams.add(p["away_team"])

    # --- Tier (a): exact map lookup ---
    home_torvik = _torvik_name_for(odds_home)
    away_torvik = _torvik_name_for(odds_away)

    if home_torvik and away_torvik:
        result = _find_prediction_by_teams(home_torvik, away_torvik, predictions)
        if result:
            return result

    # --- Tier (b): normalized matching ---
    if not home_torvik:
        home_torvik = _fuzzy_match_team(odds_home, pred_teams)
        if home_torvik:
            log.warning(
                "Fuzzy-matched Odds team %r → %r (add to TEAM_NAME_MAP)",
                odds_home,
                home_torvik,
            )
    if not away_torvik:
        away_torvik = _fuzzy_match_team(odds_away, pred_teams)
        if away_torvik:
            log.warning(
                "Fuzzy-matched Odds team %r → %r (add to TEAM_NAME_MAP)",
                odds_away,
                away_torvik,
            )

    if home_torvik and away_torvik:
        result = _find_prediction_by_teams(home_torvik, away_torvik, predictions)
        if result:
            return result

    # --- Tier (c): substring fallback (try swapping home/away too) ---
    if not home_torvik:
        home_torvik = _substring_match(odds_home, pred_teams)
        if home_torvik:
            log.warning(
                "Substring-matched Odds team %r → %r (add to TEAM_NAME_MAP)",
                odds_home,
                home_torvik,
            )
    if not away_torvik:
        away_torvik = _substring_match(odds_away, pred_teams)
        if away_torvik:
            log.warning(
                "Substring-matched Odds team %r → %r (add to TEAM_NAME_MAP)",
                odds_away,
                away_torvik,
            )

    if home_torvik and away_torvik:
        result = _find_prediction_by_teams(home_torvik, away_torvik, predictions)
        if result:
            return result

    return None


def _substring_match(odds_name: str, pred_teams: set[str]) -> str | None:
    """Last-resort substring matching – longest match wins.

    Forward matches (Torvik name found inside the odds name) are preferred
    over reverse matches (odds name found inside a Torvik name).
    """
    norm = normalize_team_name(odds_name)
    norm_cmp = _normalize_for_cmp(norm)

    # Forward: Torvik name appears as whole words inside the odds name
    best: str | None = None
    best_len = 0
    for t in pred_teams:
        t_cmp = _normalize_for_cmp(t)
        if re.search(r"\b" + re.escape(t_cmp) + r"\b", norm_cmp) and len(t_cmp) > best_len:
            best = t
            best_len = len(t_cmp)
    if best:
        return best

    # Reverse: odds name appears as whole words inside a Torvik name
    # (e.g. "Omaha" found in "Nebraska Omaha").  Prefer the shortest
    # Torvik name that contains the odds name.
    best_rev: str | None = None
    best_rev_len = float("inf")
    for t in pred_teams:
        t_cmp = _normalize_for_cmp(t)
        if re.search(r"\b" + re.escape(norm_cmp) + r"\b", t_cmp) and len(t_cmp) < best_rev_len:
            best_rev = t
            best_rev_len = len(t_cmp)
    return best_rev


# ---------------------------------------------------------------------------
# 4.  Batch matching
# ---------------------------------------------------------------------------


def match_all_events(
    odds_events: list[dict],
    predictions: list[dict],
) -> dict[str, dict]:
    """Match all Odds API events to model predictions.

    Parameters
    ----------
    odds_events
        Each dict must contain ``id``, ``home_team``, ``away_team``.
    predictions
        List of prediction dicts with Torvik-keyed ``home_team``/``away_team``.

    Returns
    -------
    dict[str, dict]
        Mapping of ``odds_event["id"]`` → matched prediction dict.
        Events with no match are omitted and logged.
    """
    matched: dict[str, dict] = {}
    unmatched: list[str] = []

    for event in odds_events:
        event_id = event["id"]
        pred = match_event_to_prediction(event, predictions)
        if pred is not None:
            matched[event_id] = pred
        else:
            label = f"{event['away_team']} @ {event['home_team']}"
            unmatched.append(label)

    if unmatched:
        log.warning(
            "Could not match %d event(s) to predictions:\n  %s",
            len(unmatched),
            "\n  ".join(unmatched),
        )

    log.info("Matched %d / %d events to predictions", len(matched), len(odds_events))
    return matched

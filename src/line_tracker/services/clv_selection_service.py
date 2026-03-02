"""Step 1 – CLV-driven selection engine.

Queries historical closed rec_snapshots to learn which (market, tier,
alpha_label) combinations historically produce CLV > 0.  Returns a
"CLV filter profile" that downstream pruning can use as a gate.
"""

from __future__ import annotations

# Default: a group must have beaten the close >= 55% of the time to pass.
DEFAULT_MIN_PCT_POSITIVE = 55.0
# Minimum closed samples for the group to be considered reliable.
DEFAULT_MIN_SAMPLE = 5


def build_clv_filter_profile(
    store,
    *,
    min_pct_positive: float = DEFAULT_MIN_PCT_POSITIVE,
    min_sample: int = DEFAULT_MIN_SAMPLE,
) -> dict:
    """Build a CLV filter profile from historical closed rec_snapshots.

    Returns a nested dict keyed by ``(market, tier, alpha_label)`` with
    ``{"pct_positive": float, "avg_clv_implied": float, "count": int,
      "passes": bool}``.
    """
    closed = store.get_closed_snapshots()
    if not closed:
        return {"groups": {}, "total_closed": 0}

    # Group by (market, tier, alpha_label)
    groups: dict[tuple, list[dict]] = {}
    for snap in closed:
        key = (
            snap.get("market", ""),
            snap.get("tier", ""),
            snap.get("alpha_label", ""),
        )
        groups.setdefault(key, []).append(snap)

    profile: dict[tuple, dict] = {}
    for key, snaps in groups.items():
        cnt = len(snaps)
        positive = sum(
            1 for s in snaps
            if (s.get("clv_delta_implied") or 0) > 0
        )
        pct_pos = 100.0 * positive / cnt if cnt else 0.0
        avg_clv = (
            sum(s.get("clv_delta_implied") or 0 for s in snaps) / cnt
            if cnt else 0.0
        )
        passes = cnt >= min_sample and pct_pos >= min_pct_positive
        profile[key] = {
            "pct_positive": round(pct_pos, 1),
            "avg_clv_implied": round(avg_clv, 6),
            "count": cnt,
            "passes": passes,
        }

    return {"groups": profile, "total_closed": len(closed)}


def passes_clv_filter(entry: dict, profile: dict) -> bool:
    """Check whether *entry* passes the CLV filter from *profile*.

    If no profile data exists for the entry's group, it fails by default
    (conservative: unknown group is not trusted).
    """
    groups = profile.get("groups", {})
    if not groups:
        # No historical data yet — let everything through.
        return True

    key = (
        entry.get("market", ""),
        entry.get("tier", ""),
        entry.get("alpha_label", ""),
    )
    group = groups.get(key)
    if group is None:
        return False
    return group["passes"]

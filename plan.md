# CLV Implementation Audit

## (a) What Is Currently Correct

1. **Odds conversion math is solid.** `bet_slip.py` has correct `american_to_decimal`, `decimal_to_american`, and `implied_prob_from_american` functions that can be reused for CLV calculations.

2. **`compute_standouts` uses implied probability correctly** for live shopping-value edge vs. median — a closely related concept to CLV.

3. **Bet legs record pick details at placement time.** The `Bet` model in `bet_history.py` properly stores each leg's sportsbook, market, selection, line value, odds, and `fetched_at` timestamp.

4. **Storage layer timestamps lines on save.** `LineStore.save_lines()` persists each `BettingLine.timestamp`, giving us raw material for historical lookups.

---

## (b) What Is Wrong / Missing / Risky

### CRITICAL: No CLV implementation exists at all

There is no `best_bets.py` file, no `test_clv.py` file, and zero CLV-related code anywhere in the codebase. Everything below must be built from scratch.

### Issue 1: `commence_time` is dropped by the scraper
- **File:** `scraper.py:74` — `_parse_events` iterates over `event` dicts that contain `"commence_time"` (confirmed in test fixture at `test_scraper.py:14`), but **never reads or stores it**.
- **Impact:** Without `commence_time` on each line, there is no way to determine "last snapshot before game start" — the entire foundation of closing-line identification.

### Issue 2: `BettingLine` model has no `commence_time` field
- **File:** `models.py` — The dataclass has `timestamp` but no field for when the game starts.
- **Impact:** Even if the scraper captured it, there's nowhere to put it.

### Issue 3: Database schema lacks `commence_time` column
- **File:** `storage.py:27-43` — The `lines` table has no `commence_time` column.
- **Impact:** Historical lines cannot be queried relative to game start time.

### Issue 4: No "closing line" query method
- **File:** `storage.py` — `get_latest_for_event` returns the latest line period, not the latest line *before a cutoff time*. There is no `get_close_lines(event, bet_type, before_time)` method.

### Issue 5: `Bet` model has no `commence_time` or CLV fields
- **File:** `bet_history.py:20-33` — `Bet` stores `created_at` but not:
  - `commence_time` (needed to look up closing lines later)
  - `clv_price_prob` / `clv_decimal` (the computed CLV values)
  - `close_estimated` flag
  - `book_odds_close_*` / `best_odds_close_*`

### Issue 6: No spread/total line-value matching for CLV
- When looking up a closing line for a spread/total bet, the snapshot must match on both the **line value** (e.g., -2.5) **and** the selection (Home/Away or Over/Under). The current storage has no query support for this.

### Issue 7: No beat-close tolerance or UI labels
- No threshold constant for classifying "beat the close" vs. "matched the close" vs. "lost to the close."
- The dashboard has no CLV display anywhere.

---

## (c) Patch Plan

### Step 1: Add `commence_time` to the data model and scraper

**models.py:**
- Add `commence_time: datetime | None = None` field to `BettingLine`.

**scraper.py:**
- In `_parse_events`, read `event["commence_time"]` and pass it through to each `_parse_*` function.
- Store it on each `BettingLine` produced.

**storage.py:**
- Add `commence_time TEXT` column to the `lines` table (with migration: `ALTER TABLE lines ADD COLUMN commence_time TEXT`).
- Update `save_lines` to include `commence_time`.
- Update `_row_to_line` to read it back.

### Step 2: Add closing-line query to `LineStore`

**storage.py — new method `get_close_lines`:**
```python
def get_close_lines(
    self,
    event: str,
    bet_type: BetType,
    before: datetime,
    sportsbook: str | None = None,
) -> list[BettingLine]:
    """Get the last snapshot per sportsbook for event+bet_type where timestamp <= before."""
```
- SQL: `SELECT * FROM lines WHERE event=? AND bet_type=? AND timestamp<=? AND id IN (SELECT MAX(id) ... GROUP BY sportsbook)`
- Optional `sportsbook` filter for same-book close.

### Step 3: Create `best_bets.py` with CLV computation

**New file `src/line_tracker/best_bets.py`:**

```python
CLV_TOLERANCE = 0.001  # implied-prob threshold for "beat" vs "matched"

def compute_clv(pick_odds, close_odds) -> dict:
    """Compute CLV for a single pick vs. close.

    Returns:
        clv_price_prob: p_close - p_pick (positive = beat the close)
        clv_decimal: close_decimal - pick_decimal
        classification: "beat" | "matched" | "lost"
    """

def enrich_bet_with_clv(bet, store) -> dict:
    """Look up closing lines and compute CLV for each leg.

    For each leg:
      1. Determine commence_time from stored lines
      2. Query best_odds_close (best across all books before commence)
      3. Query book_odds_close (same book's last line before commence)
      4. For spread/total: match on line_value + selection
      5. Compute clv_price_prob and clv_decimal for both
      6. Set close_estimated=True if no line found with timestamp <= commence_time
    """
```

Key design decisions:
- **Primary metric:** `clv_price_prob = p_close - p_pick` where `p = 1 / decimal_odds` (implied probability from decimal). Positive = bettor got better odds than close.
- **Secondary metric:** `clv_decimal = decimal_close - decimal_pick`. Positive = bettor got higher decimal odds.
- **Both best-market and same-book CLV** are computed. Best-market is the primary; same-book is supplementary context.
- **Spread/total matching:** Close lookup must filter on `home_value` (spread line) or `home_value`/`away_value` (total line) matching the pick's line value, AND the correct selection side.

### Step 4: Extend `Bet` model for CLV storage

**bet_history.py:**
- Add to `Bet` dataclass:
  - `commence_time: str | None = None`
  - `clv: list[dict] | None = None` — per-leg CLV data
- Each leg CLV dict:
  ```python
  {
      "best_close_odds": float | None,
      "book_close_odds": float | None,
      "clv_price_prob_best": float | None,   # vs best market close
      "clv_price_prob_book": float | None,    # vs same-book close
      "clv_decimal_best": float | None,
      "clv_decimal_book": float | None,
      "classification_best": str,  # "beat" | "matched" | "lost" | "no_close"
      "classification_book": str,
      "close_estimated": bool,
  }
  ```
- On `settle_bet`, call `enrich_bet_with_clv` to populate CLV data.

### Step 5: Add CLV display to dashboard

**dashboard.py — `_bet_history_dialog`:**
- In the settled bets section, show per-leg CLV:
  - Label: "CLV (prob)" with value like "+1.2%" or "-0.5%"
  - Clearly mark units: "prob pts" for `clv_price_prob`, "decimal" for `clv_decimal`
  - Color: green for beat (> tolerance), gray for matched, red for lost
  - If `close_estimated`, show a warning icon/tooltip
  - Show both "vs market" and "vs same book" when available
- Add summary metrics to settled bets: average CLV, % of bets that beat the close.

### Step 6: Write `tests/test_clv.py`

Test cases covering:

1. **Favorite sign correctness:** Pick home at -150, close at -170 → bettor got better price → positive `clv_price_prob`.
2. **Underdog sign correctness:** Pick away at +130, close at +120 → bettor got better price → positive `clv_price_prob`.
3. **Close time selection:** Lines at T-2h, T-1h, T+30m with commence at T → should pick T-1h line, not T+30m.
4. **Same-book vs best-book close:** Verify both are computed independently from the correct sportsbook filter.
5. **Spread line-value matching:** Pick spread -2.5 → close lookup must match on -2.5, not -3.0 if the line moved.
6. **Total line-value matching:** Pick Over 47.5 → close lookup must match on 47.5 total.
7. **Tolerance classification:** clv_price_prob of 0.0005 → "matched", 0.002 → "beat", -0.002 → "lost".
8. **No close data available:** Verify `close_estimated=True` and `classification="no_close"`.
9. **Edge case: pick AT the close** — identical odds should classify as "matched".

### Implementation Order

1. **Step 1** (model + scraper + storage schema) — foundational, everything depends on it
2. **Step 2** (close-line query) — needed before CLV can be computed
3. **Step 3** (best_bets.py) — core CLV logic
4. **Step 6** (tests) — written alongside Step 3
5. **Step 4** (Bet model extension) — integration with bet history
6. **Step 5** (dashboard display) — UI last

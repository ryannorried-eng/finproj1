"""Core line tracking functionality."""

from line_tracker.models import BettingLine, BetType


class LineTracker:
    """Tracks and compares betting lines across sportsbooks."""

    def __init__(self):
        self._lines: list[BettingLine] = []

    def add_line(self, line: BettingLine) -> None:
        """Record a betting line."""
        self._lines.append(line)

    def get_lines_for_event(
        self, event: str, bet_type: BetType | None = None
    ) -> list[BettingLine]:
        """Get all tracked lines for a given event, optionally filtered by bet type."""
        results = [ln for ln in self._lines if ln.event == event]
        if bet_type is not None:
            results = [ln for ln in results if ln.bet_type == bet_type]
        return results

    def get_best_moneyline(
        self, event: str, side: str = "home"
    ) -> BettingLine | None:
        """Find the best (highest) moneyline odds across sportsbooks for a side."""
        lines = self.get_lines_for_event(event, BetType.MONEYLINE)
        if not lines:
            return None
        if side == "home":
            return max(lines, key=lambda ln: ln.home_value)
        return max(lines, key=lambda ln: ln.away_value)

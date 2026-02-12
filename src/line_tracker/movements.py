"""Line movement detection — flags significant changes over time."""

from __future__ import annotations

from dataclasses import dataclass

from line_tracker.models import BettingLine, BetType


@dataclass
class LineMove:
    """A detected line movement."""

    event: str
    sportsbook: str
    bet_type: BetType
    old_line: BettingLine
    new_line: BettingLine
    change: float  # absolute change in the primary value

    @property
    def direction(self) -> str:
        if self.change > 0:
            return "up"
        elif self.change < 0:
            return "down"
        return "unchanged"


def detect_moves(
    old_lines: list[BettingLine],
    new_lines: list[BettingLine],
    threshold: float = 0.0,
) -> list[LineMove]:
    """Compare two snapshots and return significant movements.

    Args:
        old_lines: Previous snapshot of lines.
        new_lines: Current snapshot of lines.
        threshold: Minimum absolute change to report. 0 = report all changes.

    Returns:
        List of LineMove objects for lines that changed.
    """
    old_map = _index_lines(old_lines)
    moves: list[LineMove] = []

    for ln in new_lines:
        key = _line_key(ln)
        prev = old_map.get(key)
        if prev is None:
            continue

        change = _compute_change(prev, ln)
        if abs(change) > threshold:
            moves.append(
                LineMove(
                    event=ln.event,
                    sportsbook=ln.sportsbook,
                    bet_type=ln.bet_type,
                    old_line=prev,
                    new_line=ln,
                    change=round(change, 2),
                )
            )

    moves.sort(key=lambda m: abs(m.change), reverse=True)
    return moves


def _line_key(ln: BettingLine) -> tuple:
    """Unique key for matching lines across snapshots."""
    return (ln.event, ln.sportsbook, ln.bet_type)


def _index_lines(
    lines: list[BettingLine],
) -> dict[tuple, BettingLine]:
    """Index lines by their unique key (keeps latest per key)."""
    index: dict[tuple, BettingLine] = {}
    for ln in lines:
        index[_line_key(ln)] = ln
    return index


def _compute_change(old: BettingLine, new: BettingLine) -> float:
    """Compute the primary value change based on bet type."""
    if old.bet_type == BetType.MONEYLINE:
        return new.home_value - old.home_value
    elif old.bet_type == BetType.SPREAD:
        return new.home_value - old.home_value
    elif old.bet_type == BetType.TOTAL:
        return new.home_value - old.home_value
    return 0.0

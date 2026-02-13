"""Alert system for line movements and arbitrage opportunities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from line_tracker.arbitrage import ArbOpportunity
from line_tracker.models import BetType
from line_tracker.movements import LineMove


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A triggered alert."""

    level: AlertLevel
    message: str
    source: str  # "movement" or "arbitrage"


@dataclass
class AlertManager:
    """Configurable alert engine for line moves and arb opportunities."""

    move_threshold: float = 10.0  # ML odds change to trigger
    spread_threshold: float = 0.5  # spread point change to trigger
    total_threshold: float = 0.5  # total point change to trigger
    arb_min_margin: float = 0.0  # min arb margin % to trigger
    callbacks: list = field(default_factory=list)

    def check_movements(self, moves: list[LineMove]) -> list[Alert]:
        """Generate alerts from line movements."""
        alerts: list[Alert] = []
        for move in moves:
            threshold = self._threshold_for(move)
            if abs(move.change) < threshold:
                continue

            level = self._level_for_move(move, threshold)
            msg = (
                f"{move.sportsbook} {move.event} "
                f"{move.bet_type.value}: "
                f"{move.old_line.home_value} -> "
                f"{move.new_line.home_value} "
                f"({move.change:+.1f})"
            )
            alert = Alert(
                level=level, message=msg, source="movement"
            )
            alerts.append(alert)
            self._notify(alert)

        return alerts

    def check_arbitrage(
        self, arbs: list[ArbOpportunity]
    ) -> list[Alert]:
        """Generate alerts from arbitrage opportunities."""
        alerts: list[Alert] = []
        for arb in arbs:
            if arb.margin < self.arb_min_margin:
                continue

            if arb.profitable:
                level = AlertLevel.CRITICAL
            else:
                level = AlertLevel.INFO

            msg = (
                f"ARB {arb.event} {arb.bet_type.value}: "
                f"{arb.side_a.sportsbook} vs "
                f"{arb.side_b.sportsbook} "
                f"(margin: {arb.margin:+.2f}%)"
            )
            alert = Alert(
                level=level, message=msg, source="arbitrage"
            )
            alerts.append(alert)
            self._notify(alert)

        return alerts

    def _threshold_for(self, move: LineMove) -> float:
        if move.bet_type == BetType.SPREAD:
            return self.spread_threshold
        if move.bet_type == BetType.TOTAL:
            return self.total_threshold
        return self.move_threshold

    def _level_for_move(
        self, move: LineMove, threshold: float
    ) -> AlertLevel:
        magnitude = abs(move.change)
        if magnitude >= threshold * 3:
            return AlertLevel.CRITICAL
        if magnitude >= threshold * 2:
            return AlertLevel.WARNING
        return AlertLevel.INFO

    def _notify(self, alert: Alert) -> None:
        for callback in self.callbacks:
            callback(alert)

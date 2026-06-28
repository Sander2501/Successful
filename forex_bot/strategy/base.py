"""Strategy interface.

A strategy is a pure decision function: it receives the just-closed candle plus
a read-only :class:`StrategyContext` and emits a :class:`Signal` (intent). It
must not place orders, log to brokers, or persist state — those are side
effects handled by the execution and risk layers. This keeps strategies
identical between backtest and live, and trivially unit-testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import Candle, Position, Signal


@dataclass
class StrategyContext:
    """Read-only view passed to a strategy on each candle.

    ``history`` is the rolling buffer of recently-closed candles for the current
    instrument (most recent last), *including* the candle being evaluated.
    """

    history: Sequence[Candle]
    position: Position | None = None
    equity: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def closes(self) -> list[float]:
        return [c.close for c in self.history]

    @property
    def highs(self) -> list[float]:
        return [c.high for c in self.history]

    @property
    def lows(self) -> list[float]:
        return [c.low for c in self.history]


class StrategyBase(ABC):
    """Base class for all strategies."""

    #: Minimum number of candles needed before signals are meaningful.
    warmup: int = 1

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def on_candle(self, candle: Candle, context: StrategyContext) -> Signal | None:
        """Return a Signal (intent) or ``None`` to do nothing."""
        raise NotImplementedError

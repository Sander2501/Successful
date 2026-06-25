"""RSI mean-reversion strategy.

Enters long when RSI exits oversold territory and short when it exits
overbought, and exits the position when RSI returns to a neutral band. A second
baseline strategy that demonstrates the pluggable interface and exit signals.
"""

from __future__ import annotations

from typing import Optional

from ..indicators import rsi
from ..models import Candle, Signal, SignalType
from .base import StrategyBase, StrategyContext


class RsiReversionStrategy(StrategyBase):
    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level
        self.warmup = period + 2

    def on_candle(self, candle: Candle, context: StrategyContext) -> Optional[Signal]:
        closes = context.closes
        if len(closes) < self.warmup:
            return None

        series = rsi(closes, self.period)
        now, prev = series[-1], series[-2]
        if now is None or prev is None:
            return None

        has_position = context.position is not None

        # Exit logic first: close when RSI reverts through the neutral band.
        if has_position:
            side = context.position.side
            if side.value == "BUY" and prev < self.exit_level <= now:
                return Signal(candle.epic, SignalType.EXIT, candle.timestamp,
                              meta={"rsi": now, "reason": "revert_up"})
            if side.value == "SELL" and prev > self.exit_level >= now:
                return Signal(candle.epic, SignalType.EXIT, candle.timestamp,
                              meta={"rsi": now, "reason": "revert_down"})
            return None

        # Entry logic: cross back out of an extreme.
        if prev <= self.oversold < now:
            return Signal(candle.epic, SignalType.ENTER_LONG, candle.timestamp,
                          meta={"rsi": now, "reason": "exit_oversold"})
        if prev >= self.overbought > now:
            return Signal(candle.epic, SignalType.ENTER_SHORT, candle.timestamp,
                          meta={"rsi": now, "reason": "exit_overbought"})
        return None

"""EMA crossover trend-following strategy.

Goes long when the fast EMA crosses above the slow EMA, short on the reverse
cross, and uses an ATR-based protective stop. A baseline, interpretable
strategy used to exercise the full data -> strategy -> risk -> execution path.
"""

from __future__ import annotations

from typing import Optional

from ..indicators import atr, ema
from ..models import Candle, Signal, SignalType
from .base import StrategyBase, StrategyContext


class EmaCrossoverStrategy(StrategyBase):
    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
    ) -> None:
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.warmup = slow + 1

    def on_candle(self, candle: Candle, context: StrategyContext) -> Optional[Signal]:
        closes = context.closes
        if len(closes) < self.warmup:
            return None

        fast_ema = ema(closes, self.fast)
        slow_ema = ema(closes, self.slow)
        f_now, f_prev = fast_ema[-1], fast_ema[-2]
        s_now, s_prev = slow_ema[-1], slow_ema[-2]
        if None in (f_now, f_prev, s_now, s_prev):
            return None

        crossed_up = f_prev <= s_prev and f_now > s_now
        crossed_down = f_prev >= s_prev and f_now < s_now
        if not (crossed_up or crossed_down):
            return None

        atr_series = atr(context.highs, context.lows, closes, self.atr_period)
        atr_now = atr_series[-1] if atr_series and atr_series[-1] is not None else None
        price = candle.close

        if crossed_up:
            sl = price - self.atr_stop_mult * atr_now if atr_now else None
            tp = price + self.atr_target_mult * atr_now if atr_now else None
            return Signal(
                epic=candle.epic,
                type=SignalType.ENTER_LONG,
                timestamp=candle.timestamp,
                stop_loss=sl,
                take_profit=tp,
                meta={"fast_ema": f_now, "slow_ema": s_now, "atr": atr_now},
            )
        # crossed_down
        sl = price + self.atr_stop_mult * atr_now if atr_now else None
        tp = price - self.atr_target_mult * atr_now if atr_now else None
        return Signal(
            epic=candle.epic,
            type=SignalType.ENTER_SHORT,
            timestamp=candle.timestamp,
            stop_loss=sl,
            take_profit=tp,
            meta={"fast_ema": f_now, "slow_ema": s_now, "atr": atr_now},
        )

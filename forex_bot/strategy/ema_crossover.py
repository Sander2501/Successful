"""EMA crossover trend-following strategy.

Goes long when the fast EMA crosses above the slow EMA, short on the reverse
cross, with an ATR-based protective stop and target.

Two optional, **stateless** filters reduce the whipsaw that plagues a bare
crossover in ranging markets:

  * ``trend_filter`` — a long-period EMA acting as a regime gate. Longs are only
    taken when price is above it (and shorts below), so the strategy trades with
    the dominant trend instead of fighting chop.
  * ``min_separation_pct`` — require the two EMAs to be at least this far apart
    (as a fraction of price) at the cross, discarding marginal crosses where the
    averages are effectively tangled.

Both default to off so behaviour is unchanged unless configured; the example
config enables them.
"""

from __future__ import annotations

from ..indicators import adx, atr, ema
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
        trend_filter: int | None = None,
        min_separation_pct: float = 0.0,
        adx_period: int | None = None,
        adx_threshold: float = 20.0,
    ) -> None:
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        if trend_filter is not None and trend_filter <= slow:
            raise ValueError("trend_filter period should be larger than the slow period")
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.trend_filter = trend_filter
        self.min_separation_pct = min_separation_pct
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.warmup = max(slow, trend_filter or 0, (adx_period or 0) * 2) + 1

    def on_candle(self, candle: Candle, context: StrategyContext) -> Signal | None:
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

        price = candle.close

        # Filter 1: marginal crosses. At a cross the two EMAs are by definition
        # nearly equal, so a bare "are they apart now?" test is almost useless.
        # Require the gap to clear the threshold AND be *widening* through the
        # cross (momentum into it), which is what distinguishes a decisive cross
        # from the EMAs tangling sideways.
        if self.min_separation_pct > 0 and price > 0:
            sep_now = abs(f_now - s_now)
            sep_prev = abs(f_prev - s_prev)
            if sep_now / price < self.min_separation_pct or sep_now <= sep_prev:
                return None

        # Filter 2: regime gate — only trade in the direction of the long EMA.
        trend_dir = self._trend_direction(closes)
        if trend_dir is not None:
            if crossed_up and trend_dir < 0:
                return None
            if crossed_down and trend_dir > 0:
                return None

        # Filter 3: ADX trend-strength gate — skip entries in non-trending regimes.
        if self.adx_period is not None and not self._is_trending(context):
            return None

        atr_series = atr(context.highs, context.lows, closes, self.atr_period)
        atr_now = atr_series[-1] if atr_series and atr_series[-1] is not None else None

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

    def _trend_direction(self, closes: list[float]) -> int | None:
        """+1 if price is above the trend EMA, -1 if below, None if disabled."""
        if self.trend_filter is None:
            return None
        trend = ema(closes, self.trend_filter)
        t_now = trend[-1]
        if t_now is None:
            return None
        return 1 if closes[-1] >= t_now else -1

    def _is_trending(self, context: StrategyContext) -> bool:
        """True when ADX confirms a trending regime.

        Fails CLOSED: if ADX can't be computed yet, treat the regime as unknown
        and skip the trade rather than waving it through (the old behavior, which
        let entries pass whenever the gate had no reading — an optimistic bias).
        """
        series = adx(context.highs, context.lows, context.closes, self.adx_period)
        value = series[-1] if series else None
        if value is None:
            return False
        return value >= self.adx_threshold

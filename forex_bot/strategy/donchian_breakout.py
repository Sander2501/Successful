"""Donchian channel breakout - a classic trend/momentum system.

Enter long when price breaks above the highest high of the last ``entry`` bars,
short when it breaks below the lowest low; exit when price crosses the opposite
``exit`` channel. An ADX regime gate (on by default) suppresses breakouts in
non-trending markets, which is where breakout systems bleed the most. Protective
stops are ATR-based.

This is the well-documented "Turtle"-style breakout. Time-series momentum of
this kind is one of the more robust cross-asset anomalies, which makes it a
sensible second strategy to validate out-of-sample rather than a curve fit.
"""

from __future__ import annotations

from typing import Optional

from ..indicators import adx, atr, donchian
from ..models import Candle, Signal, SignalType
from .base import StrategyBase, StrategyContext


class DonchianBreakoutStrategy(StrategyBase):
    def __init__(
        self,
        entry: int = 20,
        exit: int = 10,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        adx_period: Optional[int] = 14,
        adx_threshold: float = 20.0,
        atr_regime_lookback: Optional[int] = None,
        atr_regime_quantile: float = 0.6,
    ) -> None:
        if entry < 2:
            raise ValueError("entry period must be >= 2")
        if exit < 1 or exit >= entry:
            raise ValueError("exit period must satisfy 1 <= exit < entry")
        if atr_regime_lookback is not None and atr_regime_lookback < 2:
            raise ValueError("atr_regime_lookback must be >= 2 when enabled")
        if not 0.0 <= atr_regime_quantile <= 1.0:
            raise ValueError("atr_regime_quantile must be between 0 and 1")
        self.entry = entry
        self.exit = exit
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_regime_lookback = atr_regime_lookback
        self.atr_regime_quantile = atr_regime_quantile
        regime_warmup = atr_period + (atr_regime_lookback or 0)
        self.warmup = max(entry, (adx_period or 0) * 2, atr_period, regime_warmup) + 2

    def on_candle(self, candle: Candle, context: StrategyContext) -> Optional[Signal]:
        highs, lows, closes = context.highs, context.lows, context.closes
        if len(closes) < self.warmup:
            return None

        # Channels computed on bars *excluding* the current one (no look-ahead):
        # breakout is the current close versus the prior-window extremes.
        entry_up, entry_lo = donchian(highs[:-1], lows[:-1], self.entry)
        exit_up, exit_lo = donchian(highs[:-1], lows[:-1], self.exit)
        upper, lower = entry_up[-1], entry_lo[-1]
        x_upper, x_lower = exit_up[-1], exit_lo[-1]
        if None in (upper, lower, x_upper, x_lower):
            return None

        price = candle.close
        position = context.position

        # Exit management for an existing position (opposite short-channel break).
        if position is not None:
            if position.side.value == "BUY" and price < x_lower:
                return Signal(candle.epic, SignalType.EXIT, candle.timestamp,
                              meta={"reason": "exit_channel"})
            if position.side.value == "SELL" and price > x_upper:
                return Signal(candle.epic, SignalType.EXIT, candle.timestamp,
                              meta={"reason": "exit_channel"})
            return None

        breakout_up = price > upper
        breakout_down = price < lower
        if not (breakout_up or breakout_down):
            return None

        # Regime gate: only take breakouts when ADX confirms a trend.
        if self.adx_period is not None:
            series = adx(highs, lows, closes, self.adx_period)
            value = series[-1] if series else None
            if value is not None and value < self.adx_threshold:
                return None

        atr_series = atr(highs, lows, closes, self.atr_period)
        atr_now = atr_series[-1] if atr_series and atr_series[-1] is not None else None
        atr_threshold = None
        if self.atr_regime_lookback is not None:
            atr_threshold = _atr_regime_threshold(
                atr_series,
                self.atr_regime_lookback,
                self.atr_regime_quantile,
            )
            if atr_now is None or atr_threshold is None or atr_now < atr_threshold:
                return None

        if breakout_up:
            sl = price - self.atr_stop_mult * atr_now if atr_now else None
            return Signal(candle.epic, SignalType.ENTER_LONG, candle.timestamp,
                          stop_loss=sl, meta={
                              "channel_high": upper,
                              "atr": atr_now,
                              "atr_regime_threshold": atr_threshold,
                          })
        sl = price + self.atr_stop_mult * atr_now if atr_now else None
        return Signal(candle.epic, SignalType.ENTER_SHORT, candle.timestamp,
                      stop_loss=sl, meta={
                          "channel_low": lower,
                          "atr": atr_now,
                          "atr_regime_threshold": atr_threshold,
                      })


def _atr_regime_threshold(
    atr_series: list[Optional[float]],
    lookback: int,
    quantile: float,
) -> Optional[float]:
    """Return a trailing ATR percentile using only bars before the current one.

    ``atr_series[-1]`` belongs to the current closed candle. The threshold is
    built from ``[-lookback-1:-1]`` so the current candle cannot lift its own
    benchmark and accidentally add a subtle look-ahead/self-reference.
    """
    window = [v for v in atr_series[-lookback - 1:-1] if v is not None]
    if len(window) < lookback:
        return None
    return _quantile(window, quantile)


def _quantile(values: list[float], quantile: float) -> float:
    """Linear-interpolated quantile for a non-empty list."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * quantile
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

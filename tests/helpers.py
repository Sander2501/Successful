"""Shared test helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from forex_bot.models import Candle


def make_candles(closes, epic="TEST", timeframe="MINUTE_15", start=None, spread=0.0005):
    """Build a list of Candles from a list of close prices.

    Highs/lows are derived from neighbouring closes so OHLC stays consistent.
    """
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        hi = max(o, c) + spread
        lo = min(o, c) - spread
        candles.append(
            Candle(epic=epic, timeframe=timeframe,
                   timestamp=start + timedelta(minutes=15 * i),
                   open=o, high=hi, low=lo, close=c, volume=100.0)
        )
        prev = c
    return candles

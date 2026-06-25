"""Pure technical-indicator functions.

All functions take a list of floats (typically closes) and return a list of the
same length, with ``None`` for positions where the indicator is not yet defined
(i.e. before enough data has accumulated). Keeping them pure and stdlib-only
makes them trivial to unit-test and identical between backtest and live.
"""

from __future__ import annotations

from typing import Optional


def sma(values: list[float], period: int) -> list[Optional[float]]:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[Optional[float]] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: list[float], period: int) -> list[Optional[float]]:
    """Exponential moving average.

    Seeded with the SMA of the first ``period`` values, which is the standard
    convention and keeps the series stable.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def rsi(values: list[float], period: int = 14) -> list[Optional[float]]:
    """Wilder's Relative Strength Index."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def true_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    """True range series (first element uses high-low only)."""
    n = len(closes)
    out = [0.0] * n
    if n == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        prev_close = closes[i - 1]
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
    return out


def atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[Optional[float]]:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(highs, lows, closes)
    out: list[Optional[float]] = [None] * len(closes)
    if len(closes) < period:
        return out
    seed = sum(tr[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(closes)):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

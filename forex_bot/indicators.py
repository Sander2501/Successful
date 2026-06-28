"""Pure technical-indicator functions.

All functions take a list of floats (typically closes) and return a list of the
same length, with ``None`` for positions where the indicator is not yet defined
(i.e. before enough data has accumulated). Keeping them pure and stdlib-only
makes them trivial to unit-test and identical between backtest and live.
"""

from __future__ import annotations


def sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average.

    Seeded with the SMA of the first ``period`` values, which is the standard
    convention and keeps the series stable.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
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


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    """Wilder's Relative Strength Index."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
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
) -> list[float | None]:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(highs, lows, closes)
    out: list[float | None] = [None] * len(closes)
    if len(closes) < period:
        return out
    seed = sum(tr[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(closes)):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def dmi(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Directional Movement Index: returns (+DI, -DI, ADX), Wilder-smoothed.

    ADX measures trend *strength* irrespective of direction; +DI/-DI give the
    direction. Used as a regime filter so trend strategies only act when a trend
    is actually present.
    """
    n = len(closes)
    plus_di: list[float | None] = [None] * n
    minus_di: list[float | None] = [None] * n
    adx_out: list[float | None] = [None] * n
    if n <= period:
        return plus_di, minus_di, adx_out

    tr = true_range(highs, lows, closes)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder running sums seeded over the first `period` deltas (indices 1..period).
    atr_s = sum(tr[1 : period + 1])
    pdm_s = sum(plus_dm[1 : period + 1])
    mdm_s = sum(minus_dm[1 : period + 1])

    dx_series: list[tuple[int, float]] = []

    def _di(p: float, m: float, t: float) -> tuple[float, float]:
        if t == 0:
            return 0.0, 0.0
        return 100.0 * p / t, 100.0 * m / t

    pdi, mdi = _di(pdm_s, mdm_s, atr_s)
    plus_di[period], minus_di[period] = pdi, mdi
    dx_series.append((period, _dx(pdi, mdi)))

    for i in range(period + 1, n):
        atr_s = atr_s - atr_s / period + tr[i]
        pdm_s = pdm_s - pdm_s / period + plus_dm[i]
        mdm_s = mdm_s - mdm_s / period + minus_dm[i]
        pdi, mdi = _di(pdm_s, mdm_s, atr_s)
        plus_di[i], minus_di[i] = pdi, mdi
        dx_series.append((i, _dx(pdi, mdi)))

    # ADX = Wilder-smoothed DX, seeded by the average of the first `period` DX values.
    if len(dx_series) >= period:
        seed_idx = dx_series[period - 1][0]
        seed = sum(d for _, d in dx_series[:period]) / period
        adx_out[seed_idx] = seed
        prev = seed
        for idx, d in dx_series[period:]:
            prev = (prev * (period - 1) + d) / period
            adx_out[idx] = prev
    return plus_di, minus_di, adx_out


def adx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[float | None]:
    """Average Directional Index (trend-strength) series."""
    return dmi(highs, lows, closes, period)[2]


def donchian(
    highs: list[float], lows: list[float], period: int
) -> tuple[list[float | None], list[float | None]]:
    """Donchian channel: (upper, lower) = rolling max-high / min-low over `period`."""
    if period <= 0:
        raise ValueError("period must be positive")
    n = len(highs)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window_h = highs[i - period + 1 : i + 1]
        window_l = lows[i - period + 1 : i + 1]
        upper[i] = max(window_h)
        lower[i] = min(window_l)
    return upper, lower


def _dx(pdi: float, mdi: float) -> float:
    total = pdi + mdi
    return 100.0 * abs(pdi - mdi) / total if total > 0 else 0.0


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

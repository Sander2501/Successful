"""Aggregate live tick/quote updates into fixed-timeframe candles.

The live engine feeds mid-price updates in; whenever a price arrives that
belongs to a new bar bucket, the previous (now-closed) candle is returned so the
strategy can act on it. This mirrors the bars produced by the historical API.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import Candle

_TIMEFRAME_SECONDS = {
    "MINUTE": 60,
    "MINUTE_5": 300,
    "MINUTE_15": 900,
    "MINUTE_30": 1800,
    "HOUR": 3600,
    "HOUR_4": 14400,
    "DAY": 86400,
    "WEEK": 604800,
}


def timeframe_to_seconds(timeframe: str) -> int:
    try:
        return _TIMEFRAME_SECONDS[timeframe]
    except KeyError:
        raise ValueError(f"unknown timeframe '{timeframe}'") from None


def _bucket_start(ts: datetime, step: int) -> datetime:
    epoch = int(ts.timestamp())
    start = epoch - (epoch % step)
    return datetime.fromtimestamp(start, tz=timezone.utc)


class CandleBuilder:
    """Builds candles for a single (epic, timeframe)."""

    def __init__(self, epic: str, timeframe: str) -> None:
        self.epic = epic
        self.timeframe = timeframe
        self.step = timeframe_to_seconds(timeframe)
        self._bucket: datetime | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._v = 0.0

    def update(self, price: float, ts: datetime | None = None,
               volume: float = 0.0) -> Candle | None:
        """Feed a price. Returns a closed Candle when a bar boundary is crossed."""
        ts = ts or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bucket = _bucket_start(ts, self.step)

        if self._bucket is None:
            self._open_bucket(bucket, price, volume)
            return None

        if bucket == self._bucket:
            self._h = max(self._h, price)
            self._l = min(self._l, price)
            self._c = price
            self._v += volume
            return None

        # New bucket: close the previous candle and start fresh.
        closed = self._closed_candle()
        self._open_bucket(bucket, price, volume)
        return closed

    def flush(self) -> Candle | None:
        """Force-close the in-progress candle (e.g. on shutdown)."""
        if self._bucket is None:
            return None
        candle = self._closed_candle()
        self._bucket = None
        return candle

    # ------------------------------------------------------------------ #
    def _open_bucket(self, bucket: datetime, price: float, volume: float) -> None:
        self._bucket = bucket
        self._o = self._h = self._l = self._c = price
        self._v = volume

    def _closed_candle(self) -> Candle:
        return Candle(
            epic=self.epic,
            timeframe=self.timeframe,
            timestamp=self._bucket,  # type: ignore[arg-type]
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            volume=self._v,
        )

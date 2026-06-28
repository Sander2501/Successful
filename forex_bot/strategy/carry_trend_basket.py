"""Lower-frequency carry/trend basket strategy.

This is intentionally different from channel breakouts: it ranks a whole FX
basket by long-horizon, volatility-adjusted trend and only holds the strongest
long candidates and weakest short candidates on a slow rebalance cadence.

Carry is optional and must be provided as a currency-yield map through params.
That keeps the backtest honest: we do not hardcode today's policy rates and
accidentally apply them to historical periods.
"""

from __future__ import annotations

from datetime import datetime
from math import sqrt

from ..indicators import atr
from ..models import Candle, Signal, SignalType
from .portfolio_base import PortfolioContext, PortfolioStrategy


class CarryTrendBasketStrategy(PortfolioStrategy):
    def __init__(
        self,
        lookback: int = 250,
        top_k: int = 1,
        bottom_k: int | None = None,
        rebalance_bars: int = 30,
        min_abs_trend: float = 0.01,
        atr_period: int = 14,
        atr_stop_mult: float = 3.0,
        carry_weight: float = 0.0,
        currency_yields: dict[str, float] | None = None,
    ) -> None:
        if lookback < 5:
            raise ValueError("lookback must be >= 5")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        bottom = top_k if bottom_k is None else bottom_k
        if bottom < 0:
            raise ValueError("bottom_k must be >= 0")
        if rebalance_bars < 1:
            raise ValueError("rebalance_bars must be >= 1")
        if min_abs_trend < 0:
            raise ValueError("min_abs_trend must be >= 0")
        if atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult must be > 0")

        self.lookback = lookback
        self.top_k = top_k
        self.bottom_k = bottom
        self.rebalance_bars = rebalance_bars
        self.min_abs_trend = min_abs_trend
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.carry_weight = carry_weight
        self.currency_yields = {k.upper(): float(v) for k, v in (currency_yields or {}).items()}
        self.warmup = max(lookback + 1, atr_period + 1)
        self._bars = 0

    def on_bar(
        self,
        timestamp: datetime,
        latest: dict[str, Candle],
        context: PortfolioContext,
    ) -> list[Signal]:
        self._bars += 1
        if self._bars % self.rebalance_bars != 0:
            return []

        epics = sorted(latest)
        if len(epics) < self.top_k + self.bottom_k:
            return []

        series = context.aligned_closes(epics, self.lookback + 1)
        if series is None:
            return []

        stats = {epic: self._score_epic(epic, closes) for epic, closes in series.items()}
        ranked = sorted(epics, key=lambda epic: stats[epic]["score"], reverse=True)

        longs: set[str] = set()
        shorts: set[str] = set()
        for epic in ranked:
            if len(longs) >= self.top_k:
                break
            if stats[epic]["trend"] >= self.min_abs_trend:
                longs.add(epic)
        for epic in reversed(ranked):
            if len(shorts) >= self.bottom_k:
                break
            if epic not in longs and stats[epic]["trend"] <= -self.min_abs_trend:
                shorts.add(epic)

        signals: list[Signal] = []
        for epic in context.positions:
            if epic not in longs and epic not in shorts:
                signals.append(Signal(epic, SignalType.EXIT, timestamp,
                                      meta={"reason": "rebalance_out"}))

        for epic in ranked:
            if epic in longs:
                signals.append(self._entry_signal(epic, SignalType.ENTER_LONG, timestamp,
                                                  latest[epic], context, stats[epic]))
            elif epic in shorts:
                signals.append(self._entry_signal(epic, SignalType.ENTER_SHORT, timestamp,
                                                  latest[epic], context, stats[epic]))
        return signals

    def _score_epic(self, epic: str, closes: list[float]) -> dict[str, float]:
        trend = closes[-1] / closes[0] - 1.0 if closes[0] else 0.0
        vol = _realized_vol(closes)
        trend_score = trend / vol if vol > 0 else 0.0
        carry = _carry_diff(epic, self.currency_yields)
        return {
            "trend": trend,
            "vol": vol,
            "carry": carry,
            "score": trend_score + self.carry_weight * carry,
        }

    def _entry_signal(
        self,
        epic: str,
        signal_type: SignalType,
        timestamp: datetime,
        candle: Candle,
        context: PortfolioContext,
        stats: dict[str, float],
    ) -> Signal:
        stop = None
        hist = context.histories.get(epic, [])
        if len(hist) >= self.atr_period:
            highs = [c.high for c in hist]
            lows = [c.low for c in hist]
            closes = [c.close for c in hist]
            atr_series = atr(highs, lows, closes, self.atr_period)
            atr_now = atr_series[-1] if atr_series and atr_series[-1] is not None else None
            if atr_now:
                if signal_type is SignalType.ENTER_LONG:
                    stop = candle.close - self.atr_stop_mult * atr_now
                else:
                    stop = candle.close + self.atr_stop_mult * atr_now
        return Signal(epic, signal_type, timestamp, stop_loss=stop, meta={
            "trend": round(stats["trend"], 5),
            "vol": round(stats["vol"], 5),
            "carry": round(stats["carry"], 5),
            "score": round(stats["score"], 5),
        })


def _realized_vol(closes: list[float]) -> float:
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1]]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return sqrt(variance)


def _carry_diff(epic: str, currency_yields: dict[str, float]) -> float:
    base, quote = _split_fx_epic(epic)
    if base is None or quote is None:
        return 0.0
    # Currency yields are expected as percentages, e.g. USD: 5.25.
    return (currency_yields.get(base, 0.0) - currency_yields.get(quote, 0.0)) / 100.0


def _split_fx_epic(epic: str) -> tuple[str | None, str | None]:
    normalized = "".join(ch for ch in epic.upper() if ch.isalpha())
    if len(normalized) < 6:
        return None, None
    return normalized[:3], normalized[3:6]

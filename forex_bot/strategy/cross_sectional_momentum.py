"""Cross-sectional (relative) momentum across a basket.

Unlike the single-instrument strategies (which all fish the same price-pattern
return source), this ranks every instrument by trailing return each rebalance
and goes long the strongest / short the weakest. The bet is *relative*: "EUR is
outperforming JPY", not "EUR will rise". Cross-sectional momentum is a
well-documented, durable anomaly across asset classes and is largely orthogonal
to mean-reversion and to single-name trend signals, which is exactly why it's
worth adding as a *different* return source rather than another pattern strategy.

It is market-neutral-ish by construction (equal longs and shorts), so a common
move that lifts the whole basket largely cancels. It needs a basket to be
meaningful — on a single instrument there is nothing to rank.
"""

from __future__ import annotations

from datetime import datetime

from ..models import Candle, Signal, SignalType
from .portfolio_base import PortfolioContext, PortfolioStrategy


class CrossSectionalMomentumStrategy(PortfolioStrategy):
    def __init__(
        self,
        lookback: int = 100,
        top_k: int = 1,
        bottom_k: int | None = None,
        rebalance_bars: int = 20,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if rebalance_bars < 1:
            raise ValueError("rebalance_bars must be >= 1")
        self.lookback = lookback
        self.top_k = top_k
        self.bottom_k = top_k if bottom_k is None else bottom_k
        self.rebalance_bars = rebalance_bars
        self.warmup = lookback + 2
        self._bars = 0  # rebalance cadence counter (reset per backtest/fold)

    def on_bar(
        self,
        timestamp: datetime,
        latest: dict[str, Candle],
        context: PortfolioContext,
    ) -> list[Signal]:
        self._bars += 1
        if self._bars % self.rebalance_bars != 0:
            return []  # only act on the rebalance cadence

        epics = sorted(latest.keys())
        if len(epics) < self.top_k + self.bottom_k:
            return []  # not enough instruments to rank a long and short sleeve

        # Return over the window needs lookback+1 aligned closes.
        series = context.aligned_closes(epics, self.lookback + 1)
        if series is None:
            return []
        momentum = {
            e: (s[-1] / s[0] - 1.0) if s[0] else 0.0
            for e, s in series.items()
        }
        ranked = sorted(epics, key=lambda e: momentum[e], reverse=True)
        longs = set(ranked[: self.top_k])
        shorts = set(ranked[-self.bottom_k :])
        # A name cannot be both (guarded by the length check above, but be safe):
        shorts -= longs

        signals: list[Signal] = []
        # Exit anything no longer in either sleeve.
        for epic in context.positions:
            if epic not in longs and epic not in shorts:
                signals.append(Signal(epic, SignalType.EXIT, timestamp,
                                      meta={"reason": "rebalance_out"}))
        # Enter/keep the target sleeves (the engine ignores same-side entries and
        # reverses opposite-side ones).
        for epic in ranked:
            if epic in longs:
                signals.append(Signal(epic, SignalType.ENTER_LONG, timestamp,
                                      meta={"mom": round(momentum[epic], 5)}))
            elif epic in shorts:
                signals.append(Signal(epic, SignalType.ENTER_SHORT, timestamp,
                                      meta={"mom": round(momentum[epic], 5)}))
        return signals

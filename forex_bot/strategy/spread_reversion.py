"""Pairs / spread mean-reversion (statistical arbitrage).

Trades the *relationship* between two cointegrated instruments rather than the
direction of either. The spread

    s_t = ln(P_a) - beta * ln(P_b)

(with beta a rolling hedge ratio) tends to mean-revert for correlated pairs such
as EUR/USD and GBP/USD. When the spread's z-score is extreme we bet on reversion:
spread rich -> short A / long B; spread cheap -> long A / short B; exit when the
z-score returns toward zero.

Why this can carry an edge where directional TA does not: it is a relative-value
view, it is roughly market-neutral (a USD move that lifts both legs largely
cancels), and it exploits a statistical property (cointegration) rather than
trying to predict price. It is **not** guaranteed to work on any given pair —
that is exactly what the walk-forward harness is for. It assumes the spread is
stationary; if the pair de-couples (regime break) the trade can bleed, so an
``exit_z`` stop on extreme divergence is included.
"""

from __future__ import annotations

import math
from datetime import datetime

from ..models import Candle, Signal, SignalType
from .portfolio_base import PortfolioContext, PortfolioStrategy


class SpreadReversionStrategy(PortfolioStrategy):
    def __init__(
        self,
        lookback: int = 100,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
    ) -> None:
        if lookback < 20:
            raise ValueError("lookback must be >= 20 for a stable z-score")
        if entry_z <= exit_z:
            raise ValueError("entry_z must be greater than exit_z")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.warmup = lookback + 2

    def on_bar(
        self,
        timestamp: datetime,
        latest: dict[str, Candle],
        context: PortfolioContext,
    ) -> list[Signal]:
        epics = sorted(latest.keys())
        if len(epics) != 2:
            return []  # this strategy trades exactly one pair
        a, b = epics

        series = context.aligned_closes([a, b], self.lookback)
        if series is None:
            return []
        z, _beta = self._zscore(series[a], series[b])
        if z is None:
            return []

        pos_a = context.positions.get(a)
        pos_b = context.positions.get(b)
        in_trade = pos_a is not None or pos_b is not None

        if in_trade:
            # Exit on reversion to the mean, or on an extreme divergence stop.
            if abs(z) <= self.exit_z or abs(z) >= self.stop_z:
                reason = "revert" if abs(z) <= self.exit_z else "stop_z"
                return [
                    Signal(a, SignalType.EXIT, timestamp, meta={"z": z, "reason": reason}),
                    Signal(b, SignalType.EXIT, timestamp, meta={"z": z, "reason": reason}),
                ]
            return []

        # Entries: spread rich (z high) -> short A / long B; cheap -> the reverse.
        if z >= self.entry_z:
            return [
                Signal(a, SignalType.ENTER_SHORT, timestamp, meta={"z": z}),
                Signal(b, SignalType.ENTER_LONG, timestamp, meta={"z": z}),
            ]
        if z <= -self.entry_z:
            return [
                Signal(a, SignalType.ENTER_LONG, timestamp, meta={"z": z}),
                Signal(b, SignalType.ENTER_SHORT, timestamp, meta={"z": z}),
            ]
        return []

    def _zscore(self, a: list[float], b: list[float]) -> tuple[float | None, float]:
        """Rolling hedge ratio and current spread z-score over the window."""
        log_a = [math.log(x) for x in a if x > 0]
        log_b = [math.log(x) for x in b if x > 0]
        if len(log_a) != len(a) or len(log_b) != len(b) or len(log_a) < self.lookback:
            return None, 0.0

        n = len(log_b)
        mean_b = sum(log_b) / n
        mean_a = sum(log_a) / n
        var_b = sum((x - mean_b) ** 2 for x in log_b)
        if var_b <= 0:
            return None, 0.0
        cov = sum((log_a[i] - mean_a) * (log_b[i] - mean_b) for i in range(n))
        beta = cov / var_b

        spread = [log_a[i] - beta * log_b[i] for i in range(n)]
        mean_s = sum(spread) / n
        var_s = sum((x - mean_s) ** 2 for x in spread) / (n - 1)
        std_s = math.sqrt(var_s)
        if std_s <= 0:
            return None, beta
        z = (spread[-1] - mean_s) / std_s
        return z, beta

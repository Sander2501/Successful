"""Portfolio (multi-instrument) strategy interface.

Single-instrument strategies (:class:`StrategyBase`) decide from one epic's
candles. Some edges are inherently cross-instrument — a pairs/spread reversion
trade is a view on the *relationship* between two instruments, not on either
one alone. A :class:`PortfolioStrategy` is called once per bar with the latest
candle of every instrument and may emit signals for several epics at once.

The backtester routes to the right loop based on the strategy's base class, so
both kinds share the same risk, execution, and reporting machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..models import Candle, Position, Signal


@dataclass
class PortfolioContext:
    """Read-only view passed to a portfolio strategy each bar."""

    histories: dict[str, Sequence[Candle]]
    positions: dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    params: dict = field(default_factory=dict)

    def aligned_closes(self, epics: Sequence[str], lookback: int) -> dict[str, list[float]] | None:
        """Return per-epic close series aligned on their common timestamps.

        Returns ``None`` if any epic lacks data or fewer than ``lookback`` common
        bars are available — the strategy should then do nothing.
        """
        maps = {}
        for e in epics:
            hist = self.histories.get(e)
            if not hist:
                return None
            maps[e] = {c.timestamp: c.close for c in hist}
        common = None
        for m in maps.values():
            ks = set(m)
            common = ks if common is None else (common & ks)
        if not common:
            return None
        ordered = sorted(common)[-lookback:]
        if len(ordered) < lookback:
            return None
        return {e: [maps[e][t] for t in ordered] for e in epics}


class PortfolioStrategy(ABC):
    """Base class for multi-instrument strategies."""

    warmup: int = 1

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def on_bar(
        self,
        timestamp: datetime,
        latest: dict[str, Candle],
        context: PortfolioContext,
    ) -> list[Signal]:
        """Return zero or more Signals (across instruments) for this bar."""
        raise NotImplementedError

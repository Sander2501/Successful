"""Strategy layer: pluggable, broker-agnostic signal generators."""

from __future__ import annotations

from typing import Any

from .base import StrategyBase, StrategyContext
from .donchian_breakout import DonchianBreakoutStrategy
from .ema_crossover import EmaCrossoverStrategy
from .portfolio_base import PortfolioContext, PortfolioStrategy
from .rsi_reversion import RsiReversionStrategy
from .smc import SmcMarketAnalyzer, SmcSweepReversalStrategy
from .spread_reversion import SpreadReversionStrategy

# Registry of available strategies keyed by config name. Values may be
# single-instrument (StrategyBase) or multi-instrument (PortfolioStrategy); the
# backtester routes to the correct loop based on the instance type.
STRATEGY_REGISTRY: dict[str, type] = {
    "ema_crossover": EmaCrossoverStrategy,
    "rsi_reversion": RsiReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "spread_reversion": SpreadReversionStrategy,
    "sweep_reversal": SmcSweepReversalStrategy,
}


def build_strategy(name: str, params: dict[str, Any] | None = None) -> StrategyBase:
    """Instantiate a registered strategy by name."""
    try:
        cls = STRATEGY_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        ) from None
    return cls(**(params or {}))


__all__ = [
    "StrategyBase",
    "StrategyContext",
    "PortfolioStrategy",
    "PortfolioContext",
    "EmaCrossoverStrategy",
    "RsiReversionStrategy",
    "DonchianBreakoutStrategy",
    "SpreadReversionStrategy",
    "SmcSweepReversalStrategy",
    "SmcMarketAnalyzer",
    "STRATEGY_REGISTRY",
    "build_strategy",
]

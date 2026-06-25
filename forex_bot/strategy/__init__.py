"""Strategy layer: pluggable, broker-agnostic signal generators."""

from __future__ import annotations

from typing import Any

from .base import StrategyBase, StrategyContext
from .ema_crossover import EmaCrossoverStrategy
from .rsi_reversion import RsiReversionStrategy

# Registry of available strategies keyed by config name.
STRATEGY_REGISTRY: dict[str, type[StrategyBase]] = {
    "ema_crossover": EmaCrossoverStrategy,
    "rsi_reversion": RsiReversionStrategy,
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
    "EmaCrossoverStrategy",
    "RsiReversionStrategy",
    "STRATEGY_REGISTRY",
    "build_strategy",
]

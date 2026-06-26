"""Simulated execution for backtesting.

Applies a configurable cost model:
  * half-spread on each side (buy at ask, sell at bid),
  * fixed slippage in the adverse direction,
  * flat per-trade commission.
"""

from __future__ import annotations

from typing import Optional

from ..config import CostConfig, InstrumentSpecs
from ..models import Order, Side
from .base import ExecutionEngine, Fill


class SimulatedExecution(ExecutionEngine):
    def __init__(self, costs: CostConfig, specs: Optional[InstrumentSpecs] = None) -> None:
        self.costs = costs
        self.specs = specs

    def execute(self, order: Order, *, reference_price: float) -> Fill:
        spread = self.specs.spread(order.epic) if self.specs else self.costs.spread_points
        half_spread = spread / 2.0
        slip = self.costs.slippage_points
        # Buys fill higher, sells fill lower (adverse).
        if order.side is Side.BUY:
            price = reference_price + half_spread + slip
        else:
            price = reference_price - half_spread - slip
        return Fill(
            epic=order.epic,
            side=order.side,
            size=order.size,
            price=price,
            commission=self.costs.commission_per_trade,
        )

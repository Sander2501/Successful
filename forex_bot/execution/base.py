"""Execution interface shared by simulated and live engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models import Order, Side


@dataclass
class Fill:
    """The result of executing an order."""

    epic: str
    side: Side
    size: float
    price: float  # price actually filled at (after spread/slippage)
    commission: float = 0.0
    deal_id: str | None = None


class ExecutionEngine(ABC):
    @abstractmethod
    def execute(self, order: Order, *, reference_price: float) -> Fill:
        """Execute ``order`` and return the resulting :class:`Fill`."""
        raise NotImplementedError

"""A minimal cash + positions portfolio used by the backtester.

Tracks cash, one open position per instrument, closed trades, and a sampled
equity curve. PnL is realized on close; equity marks open positions to the
latest seen price.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..config import InstrumentSpecs
from ..execution.base import Fill
from ..models import Position, Side, Trade


class Portfolio:
    def __init__(self, starting_equity: float, *, value_per_point: float = 1.0,
                 specs: Optional[InstrumentSpecs] = None) -> None:
        self.cash = starting_equity
        self.value_per_point = value_per_point
        self.specs = specs
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self._last_price: dict[str, float] = {}

    def _vpp(self, epic: str) -> float:
        return self.specs.vpp(epic) if self.specs else self.value_per_point

    # ------------------------------------------------------------------ #
    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    def position_for(self, epic: str) -> Optional[Position]:
        return self.positions.get(epic)

    def mark_price(self, epic: str, price: float) -> None:
        self._last_price[epic] = price

    def equity(self) -> float:
        total = self.cash
        for epic, pos in self.positions.items():
            price = self._last_price.get(epic, pos.entry_price)
            total += pos.unrealized_pnl(price, self._vpp(epic))
        return total

    def record_equity(self, when: datetime) -> None:
        self.equity_curve.append((when, self.equity()))

    # ------------------------------------------------------------------ #
    def open_position(self, fill: Fill, when: datetime,
                      stop_loss: float | None = None,
                      take_profit: float | None = None) -> Position:
        self.cash -= fill.commission
        pos = Position(
            epic=fill.epic,
            side=fill.side,
            size=fill.size,
            entry_price=fill.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=when,
        )
        self.positions[fill.epic] = pos
        self._last_price[fill.epic] = fill.price
        return pos

    def close_position(self, fill: Fill, when: datetime) -> Trade:
        pos = self.positions.pop(fill.epic)
        # fill.side here is the *closing* side (opposite of position).
        pnl = (fill.price - pos.entry_price) * pos.side.sign * pos.size * self._vpp(pos.epic)
        pnl -= fill.commission
        self.cash += pnl
        # Risk taken at entry, in account currency, for R-multiple reporting.
        initial_risk = (
            abs(pos.entry_price - pos.stop_loss) * pos.size * self._vpp(pos.epic)
            if pos.stop_loss is not None else 0.0
        )
        trade = Trade(
            epic=pos.epic,
            side=pos.side,
            size=pos.size,
            entry_price=pos.entry_price,
            exit_price=fill.price,
            entry_time=pos.opened_at,
            exit_time=when,
            pnl=pnl,
            fees=fill.commission,
            initial_risk=initial_risk,
        )
        self.trades.append(trade)
        self._last_price[fill.epic] = fill.price
        return trade

    def closing_side(self, epic: str) -> Side:
        return self.positions[epic].side.opposite

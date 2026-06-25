"""A minimal cash + positions portfolio used by the backtester.

Tracks cash, one open position per instrument, closed trades, and a sampled
equity curve. PnL is realized on close; equity marks open positions to the
latest seen price.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..execution.base import Fill
from ..models import Position, Side, Trade


class Portfolio:
    def __init__(self, starting_equity: float, *, value_per_point: float = 1.0) -> None:
        self.cash = starting_equity
        self.value_per_point = value_per_point
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self._last_price: dict[str, float] = {}

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
            total += pos.unrealized_pnl(price, self.value_per_point)
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
        pnl = (fill.price - pos.entry_price) * pos.side.sign * pos.size * self.value_per_point
        pnl -= fill.commission
        self.cash += pnl
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
        )
        self.trades.append(trade)
        self._last_price[fill.epic] = fill.price
        return trade

    def closing_side(self, epic: str) -> Side:
        return self.positions[epic].side.opposite

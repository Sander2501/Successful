"""Risk management: turns a strategy Signal into an approved, sized Order.

Responsibilities:
  * Position sizing from a fixed fractional risk model (risk_per_trade of equity
    divided by the stop distance), capped by a max-notional limit.
  * Enforcing portfolio limits: max concurrent positions, daily loss halt, and
    correlation-aware per-currency exposure caps.

The same RiskManager instance is shared by the backtester and the live engine so
limits are validated identically in both.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from ..config import RiskConfig
from ..models import Order, OrderType, Position, Signal, Side
from .exposure import (
    CurrencyMap,
    net_currency_exposures,
    position_contributions,
)


@dataclass
class RiskDecision:
    """Outcome of evaluating a signal against the risk framework."""

    approved: bool
    order: Optional[Order] = None
    reason: str = ""


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        *,
        value_per_point: float = 1.0,
        currency_map: Optional[CurrencyMap] = None,
    ) -> None:
        self.config = config
        self.value_per_point = value_per_point
        self.currency_map: CurrencyMap = currency_map or {}
        self._day: Optional[date] = None
        self._day_start_equity: float = 0.0
        self._halted_for_day = False

    # ------------------------------------------------------------------ #
    # daily loss tracking
    # ------------------------------------------------------------------ #
    def start_day(self, day: date, equity: float) -> None:
        self._day = day
        self._day_start_equity = equity
        self._halted_for_day = False

    def update_equity(self, day: date, equity: float) -> None:
        """Roll the day boundary and flip the halt flag if the daily loss
        limit is breached."""
        if self._day != day:
            self.start_day(day, equity)
        if self._day_start_equity > 0:
            drawdown = (self._day_start_equity - equity) / self._day_start_equity
            if drawdown >= self.config.max_daily_loss_pct:
                self._halted_for_day = True

    @property
    def halted(self) -> bool:
        return self._halted_for_day

    # ------------------------------------------------------------------ #
    # sizing & approval
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        signal: Signal,
        *,
        price: float,
        equity: float,
        positions: Sequence[Position] = (),
    ) -> RiskDecision:
        if not signal.is_entry or signal.side is None:
            return RiskDecision(False, reason="not an entry signal")

        if self._halted_for_day:
            return RiskDecision(False, reason="daily loss limit reached; trading halted")

        if len(positions) >= self.config.max_open_positions:
            return RiskDecision(
                False, reason=f"max open positions ({self.config.max_open_positions}) reached"
            )

        if equity <= 0:
            return RiskDecision(False, reason="non-positive equity")

        size = self._size_position(signal, price=price, equity=equity)
        if size <= 0:
            return RiskDecision(False, reason="computed position size is zero")

        blocked = self._check_currency_limits(signal, size, price, equity, positions)
        if blocked is not None:
            return RiskDecision(False, reason=blocked)

        order = Order(
            epic=signal.epic,
            side=signal.side,
            size=size,
            order_type=OrderType.MARKET,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        return RiskDecision(True, order=order, reason="approved")

    def _check_currency_limits(
        self,
        signal: Signal,
        size: float,
        price: float,
        equity: float,
        positions: Sequence[Position],
    ) -> Optional[str]:
        """Reject entries that would over-concentrate a single currency.

        Returns a rejection reason, or None if the order is within limits.
        """
        candidate = position_contributions(
            signal.epic, signal.side, size, price, self.currency_map,
            value_per_point=self.value_per_point,
        )
        if not candidate:
            return None  # unknown currencies -> skip currency checks

        # Net per-currency exposure cap (existing positions + the candidate).
        # A cap of 1.0 (100% of equity) rarely binds and acts as "effectively off".
        cap = self.config.max_currency_exposure_pct
        totals = net_currency_exposures(
            positions, self.currency_map, value_per_point=self.value_per_point
        )
        for ccy, amount in candidate.items():
            projected = abs(totals.get(ccy, 0.0) + amount)
            if equity > 0 and projected > cap * equity:
                return (
                    f"currency exposure cap: {ccy} would reach "
                    f"{projected / equity:.0%} of equity (cap {cap:.0%})"
                )

        # Optional cap on the number of open positions sharing a currency.
        limit = self.config.max_positions_per_currency
        if limit is not None:
            counts: dict[str, int] = {}
            for pos in positions:
                pair = self.currency_map.get(pos.epic)
                if pair:
                    for ccy in pair:
                        counts[ccy] = counts.get(ccy, 0) + 1
            for ccy in candidate:
                if counts.get(ccy, 0) >= limit:
                    return f"max positions per currency reached for {ccy} ({limit})"
        return None

    def _size_position(self, signal: Signal, *, price: float, equity: float) -> float:
        """Fixed-fractional sizing.

        If a stop is supplied, size so that hitting the stop loses
        ``risk_per_trade`` of equity. Otherwise fall back to the max-notional
        cap. The result is always clamped by the max-notional cap.
        """
        cap_notional = equity * self.config.max_position_pct
        max_size_by_cap = cap_notional / (price * self.value_per_point) if price > 0 else 0.0

        size = max_size_by_cap
        if signal.stop_loss is not None:
            stop_distance = abs(price - signal.stop_loss)
            if stop_distance > 0:
                risk_amount = equity * self.config.risk_per_trade
                size = risk_amount / (stop_distance * self.value_per_point)

        return max(0.0, min(size, max_size_by_cap))

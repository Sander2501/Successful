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
from .correlation import CorrelationModel
from .exposure import (
    CurrencyMap,
    net_currency_exposures,
    notional,
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
        self.correlation: Optional[CorrelationModel] = None
        self._day: Optional[date] = None
        self._day_start_equity: float = 0.0
        self._halted_for_day = False
        self._peak_equity: float = 0.0
        self._killed = False

    def set_correlation(self, model: Optional[CorrelationModel]) -> None:
        """Attach a correlation model used for group-exposure limits."""
        self.correlation = model

    # ------------------------------------------------------------------ #
    # daily loss tracking
    # ------------------------------------------------------------------ #
    def start_day(self, day: date, equity: float) -> None:
        self._day = day
        self._day_start_equity = equity
        self._halted_for_day = False

    def update_equity(self, day: date, equity: float) -> None:
        """Roll the day boundary; flip the daily-loss halt and the portfolio
        kill switch if their thresholds are breached."""
        if self._day != day:
            self.start_day(day, equity)
        if self._day_start_equity > 0:
            drawdown = (self._day_start_equity - equity) / self._day_start_equity
            if drawdown >= self.config.max_daily_loss_pct:
                self._halted_for_day = True

        # Portfolio kill switch on drawdown from the all-time equity peak.
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            total_dd = (self._peak_equity - equity) / self._peak_equity
            if total_dd >= self.config.max_total_drawdown_pct:
                self._killed = True

    def kill(self) -> None:
        """Manually trip the kill switch (operator action)."""
        self._killed = True

    @property
    def halted(self) -> bool:
        """True when no new entries are allowed (daily halt or kill switch)."""
        return self._halted_for_day or self._killed

    @property
    def killed(self) -> bool:
        """True when the portfolio kill switch has tripped (flatten + stop)."""
        return self._killed

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

        if self._killed:
            return RiskDecision(False, reason="portfolio kill switch active; trading halted")

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
        if blocked is None:
            blocked = self._check_correlation_limits(signal, size, price, equity, positions)
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

    def _check_correlation_limits(
        self,
        signal: Signal,
        size: float,
        price: float,
        equity: float,
        positions: Sequence[Position],
    ) -> Optional[str]:
        """Cap net directional exposure within a data-driven correlation group."""
        model = self.correlation
        if model is None:
            return None
        gid = model.group_of(signal.epic)
        if gid is None:
            return None

        # Net exposure expressed in the group's reference direction, so two
        # positively-correlated longs add while a correlated hedge offsets.
        vpp = self.value_per_point
        cand = notional(size, price, vpp) * signal.side.sign * model.sign(signal.epic)
        net = cand
        count = 0
        for pos in positions:
            if model.group_of(pos.epic) == gid:
                net += (
                    notional(pos.size, pos.entry_price, vpp)
                    * pos.side.sign
                    * model.sign(pos.epic)
                )
                count += 1

        cap = self.config.max_correlated_exposure_pct
        if equity > 0 and abs(net) > cap * equity:
            return (
                f"correlated exposure cap: group {gid} would reach "
                f"{abs(net) / equity:.0%} of equity (cap {cap:.0%})"
            )

        limit = self.config.max_positions_per_group
        if limit is not None and count >= limit:
            return f"max positions per correlated group reached (group {gid}, {limit})"
        return None

    def _size_position(self, signal: Signal, *, price: float, equity: float) -> float:
        """Compute position size under the configured sizing model.

        ``vol_target``: size so a 1-ATR adverse move costs ``vol_target_pct`` of
        equity, equalizing risk contribution across instruments of different
        volatility. Falls back to fixed-fractional when no ATR is available.

        ``fixed_fractional`` (default): if a stop is supplied, size so hitting it
        loses ``risk_per_trade`` of equity; otherwise use the max-notional cap.

        The result is always clamped to ``max_position_pct`` of equity.
        """
        cap_notional = equity * self.config.max_position_pct
        max_size_by_cap = cap_notional / (price * self.value_per_point) if price > 0 else 0.0

        size = max_size_by_cap
        atr_val = (signal.meta or {}).get("atr")

        if self.config.sizing_mode == "vol_target" and atr_val:
            risk_amount = equity * self.config.vol_target_pct
            size = risk_amount / (atr_val * self.value_per_point)
        elif signal.stop_loss is not None:
            stop_distance = abs(price - signal.stop_loss)
            if stop_distance > 0:
                risk_amount = equity * self.config.risk_per_trade
                size = risk_amount / (stop_distance * self.value_per_point)

        return max(0.0, min(size, max_size_by_cap))

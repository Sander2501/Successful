"""Candle-replay backtesting engine.

Replays historical candles in strict time order through the *same* strategy ->
risk -> execution path used in live trading. Supports one or more instruments
by interleaving their candles by timestamp.

Order-of-operations per candle (no look-ahead):
  1. Mark the instrument to the candle close and check protective SL/TP against
     the candle's high/low — close the position if hit.
  2. Hand the closed candle to the strategy to obtain a signal.
  3. EXIT signals close an open position; entry signals are sized by the risk
     manager and (if approved) opened. A reversing entry first closes the
     opposing position.
  4. Sample the equity curve.

Fills use the candle close as the reference price; the execution engine applies
spread/slippage/commission on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from ..config import InstrumentSpecs, TradingConfig
from ..execution.base import ExecutionEngine
from ..execution.simulated import SimulatedExecution
from ..logging_setup import get_logger
from ..models import Candle, Order, OrderType, Side, Signal, SignalType, Trade
from ..risk.correlation import CorrelationModel
from ..risk.exposure import build_currency_map
from ..risk.manager import RiskManager
from ..strategy.base import StrategyBase, StrategyContext
from ..strategy.portfolio_base import PortfolioContext, PortfolioStrategy
from .portfolio import Portfolio

log = get_logger(__name__)


@dataclass
class BacktestResult:
    portfolio: Portfolio
    trades: list[Trade]
    equity_curve: list[tuple]
    starting_equity: float
    signals_emitted: int = 0
    orders_filled: int = 0
    meta: dict = field(default_factory=dict)


class Backtester:
    def __init__(
        self,
        strategy: StrategyBase,
        config: TradingConfig,
        *,
        execution: Optional[ExecutionEngine] = None,
        max_history: int = 1000,
    ) -> None:
        self.strategy = strategy
        self.config = config
        # Per-instrument spread + value-per-point, so a basket that mixes price
        # scales (EUR/USD vs USD/JPY) is priced correctly instead of charging
        # every instrument the first one's spread/value.
        specs = InstrumentSpecs(config.instruments, default_spread=config.costs.spread_points)
        self.specs = specs
        self.execution = execution or SimulatedExecution(config.costs, specs)
        self.max_history = max_history
        self.risk = RiskManager(
            config.risk,
            currency_map=build_currency_map(config.instruments),
            specs=specs,
        )
        self.portfolio = Portfolio(config.starting_equity, specs=specs)
        self._history: dict[str, list[Candle]] = {}
        self._signals = 0
        self._fills = 0
        self._kill_flattened = False

    # ------------------------------------------------------------------ #
    def run(self, candles_by_epic: dict[str, list[Candle]]) -> BacktestResult:
        merged = _interleave(candles_by_epic)
        if not merged:
            raise ValueError("no candles to backtest")

        # Estimate correlations up front so group exposure limits can apply.
        threshold = self.config.risk.correlation_threshold
        if threshold is not None and len(candles_by_epic) >= 2:
            self.risk.set_correlation(
                CorrelationModel.from_candles(candles_by_epic, threshold)
            )

        self.risk.start_day(merged[0].timestamp.date(), self.portfolio.equity())

        if isinstance(self.strategy, PortfolioStrategy):
            self._run_portfolio(candles_by_epic, merged)
        else:
            for candle in merged:
                self._on_candle(candle)

        # Close any positions left open at the end of the data.
        last_ts = merged[-1].timestamp
        for epic in list(self.portfolio.positions):
            self._close(epic, self.portfolio._last_price[epic], last_ts, reason="end_of_data")
        self.portfolio.record_equity(last_ts)

        return BacktestResult(
            portfolio=self.portfolio,
            trades=self.portfolio.trades,
            equity_curve=self.portfolio.equity_curve,
            starting_equity=self.config.starting_equity,
            signals_emitted=self._signals,
            orders_filled=self._fills,
            meta={"strategy": self.strategy.name},
        )

    # ------------------------------------------------------------------ #
    def _on_candle(self, candle: Candle) -> None:
        epic = candle.epic
        buf = self._history.setdefault(epic, [])
        buf.append(candle)
        if len(buf) > self.max_history:
            del buf[0 : len(buf) - self.max_history]

        self.portfolio.mark_price(epic, candle.close)

        # 1. Protective stops / targets, evaluated against this candle's range.
        self._check_protective_levels(candle)

        # daily loss halt + kill-switch bookkeeping
        self.risk.update_equity(candle.timestamp.date(), self.portfolio.equity())

        # Kill switch: flatten everything once, then stop opening new positions.
        if self.risk.killed:
            if not self._kill_flattened:
                for ep in list(self.portfolio.positions):
                    self._close(ep, self.portfolio._last_price[ep], candle.timestamp,
                                reason="kill_switch")
                self._kill_flattened = True
                log.warning("kill switch tripped: positions flattened, trading halted")
            self.portfolio.record_equity(candle.timestamp)
            return

        # 2. Strategy decision.
        context = StrategyContext(
            history=buf,
            position=self.portfolio.position_for(epic),
            equity=self.portfolio.equity(),
            params=self.config.strategy_params,
        )
        signal = self.strategy.on_candle(candle, context)
        if signal is not None and signal.type is not SignalType.HOLD:
            self._signals += 1
            self._handle_signal(signal, candle)

        # 4. Equity sample.
        self.portfolio.record_equity(candle.timestamp)

    def _run_portfolio(self, candles_by_epic: dict[str, list[Candle]], merged: list[Candle]) -> None:
        """Bar-synchronized loop for multi-instrument (portfolio) strategies."""
        timeline = sorted({c.timestamp for c in merged})
        by_epic_ts = {
            epic: {c.timestamp: c for c in cs} for epic, cs in candles_by_epic.items()
        }
        for ts in timeline:
            # Update every instrument that printed a bar at this timestamp.
            for epic in candles_by_epic:
                candle = by_epic_ts[epic].get(ts)
                if candle is None:
                    continue
                buf = self._history.setdefault(epic, [])
                buf.append(candle)
                if len(buf) > self.max_history:
                    del buf[0 : len(buf) - self.max_history]
                self.portfolio.mark_price(epic, candle.close)
                self._check_protective_levels(candle)

            self.risk.update_equity(ts.date(), self.portfolio.equity())
            if self.risk.killed:
                if not self._kill_flattened:
                    for ep in list(self.portfolio.positions):
                        self._close(ep, self.portfolio._last_price[ep], ts, reason="kill_switch")
                    self._kill_flattened = True
                    log.warning("kill switch tripped: positions flattened, trading halted")
                self.portfolio.record_equity(ts)
                continue

            latest = {e: self._history[e][-1] for e in self._history if self._history[e]}
            context = PortfolioContext(
                histories=self._history,
                positions=dict(self.portfolio.positions),
                equity=self.portfolio.equity(),
                params=self.config.strategy_params,
            )
            signals = self.strategy.on_bar(ts, latest, context) or []
            # Process exits before entries so freed capital/exposure is available.
            for signal in sorted(signals, key=lambda s: 0 if s.type is SignalType.EXIT else 1):
                if signal.type is SignalType.HOLD or signal.epic not in latest:
                    continue
                self._signals += 1
                self._handle_signal_at(signal, signal.epic, latest[signal.epic].close, ts)
            self.portfolio.record_equity(ts)

    def _check_protective_levels(self, candle: Candle) -> None:
        pos = self.portfolio.position_for(candle.epic)
        if pos is None:
            return
        # Gap realism: when a bar OPENS beyond the stop, the order would not have
        # filled at the stop price — it fills at the (worse) open. Stops therefore
        # use the adverse of (stop, open); the execution layer adds its configured
        # slippage on top. Take-profits are limit orders, so they fill at the TP
        # level at best (no favorable-gap bonus), keeping the model conservative.
        if pos.side is Side.BUY:
            if pos.stop_loss is not None and candle.low <= pos.stop_loss:
                fill = min(pos.stop_loss, candle.open)
                self._close(candle.epic, fill, candle.timestamp, reason="stop_loss")
            elif pos.take_profit is not None and candle.high >= pos.take_profit:
                self._close(candle.epic, pos.take_profit, candle.timestamp, reason="take_profit")
        else:  # SELL
            if pos.stop_loss is not None and candle.high >= pos.stop_loss:
                fill = max(pos.stop_loss, candle.open)
                self._close(candle.epic, fill, candle.timestamp, reason="stop_loss")
            elif pos.take_profit is not None and candle.low <= pos.take_profit:
                self._close(candle.epic, pos.take_profit, candle.timestamp, reason="take_profit")

    def _handle_signal(self, signal: Signal, candle: Candle) -> None:
        self._handle_signal_at(signal, candle.epic, candle.close, candle.timestamp)

    def _handle_signal_at(self, signal: Signal, epic: str, price: float, when) -> None:
        existing = self.portfolio.position_for(epic)

        if signal.type is SignalType.EXIT:
            if existing is not None:
                self._close(epic, price, when, reason="signal_exit")
            return

        # Entry signal.
        if existing is not None:
            if existing.side == signal.side:
                return  # already in the right direction; ignore.
            # Reverse: close first, then evaluate the new entry.
            self._close(epic, price, when, reason="reverse")

        decision = self.risk.evaluate(
            signal,
            price=price,
            equity=self.portfolio.equity(),
            positions=list(self.portfolio.positions.values()),
        )
        if not decision.approved or decision.order is None:
            log.debug("signal rejected", extra={"epic": epic, "reason": decision.reason})
            return
        self._open(decision.order, price, when)

    # ------------------------------------------------------------------ #
    def _open(self, order: Order, price: float, when) -> None:
        fill = self.execution.execute(order, reference_price=price)
        self.portfolio.open_position(
            fill, when, stop_loss=order.stop_loss, take_profit=order.take_profit
        )
        self._fills += 1
        log.info(
            "open",
            extra={"epic": fill.epic, "side": fill.side.value,
                   "size": round(fill.size, 4), "price": round(fill.price, 5)},
        )

    def _close(self, epic: str, price: float, when, reason: str) -> None:
        if epic not in self.portfolio.positions:
            return
        closing = Order(
            epic=epic,
            side=self.portfolio.closing_side(epic),
            size=self.portfolio.positions[epic].size,
            order_type=OrderType.MARKET,
        )
        fill = self.execution.execute(closing, reference_price=price)
        trade = self.portfolio.close_position(fill, when)
        self._fills += 1
        log.info(
            "close",
            extra={"epic": epic, "reason": reason, "pnl": round(trade.pnl, 2),
                   "price": round(fill.price, 5)},
        )


def _interleave(candles_by_epic: dict[str, list[Candle]]) -> list[Candle]:
    """Flatten and sort candles from all instruments by timestamp.

    Ties are broken by epic so the order is deterministic.
    """
    merged: list[Candle] = []
    for candles in candles_by_epic.values():
        merged.extend(candles)
    merged.sort(key=lambda c: (c.timestamp, c.epic))
    return merged

"""Live (demo/real) trading engine.

Wires the WebSocket price stream to the same strategy -> risk -> execution path
used by the backtester:

    quotes -> CandleBuilder -> strategy.on_candle -> RiskManager -> LiveExecution

A closed candle is only produced when a bar boundary is crossed, so strategies
act on completed bars exactly as in backtest. Positions are tracked locally and
reconciled against the broker on start.

This module imports broker code lazily through the clients it is given, so it is
only used when you actually run live; the pure-stdlib core never imports it.
"""

from __future__ import annotations

from typing import Optional

from .api.rest_client import CapitalRestClient
from .api.websocket_client import CapitalWebSocketClient
from .config import TradingConfig
from .data.candle_builder import CandleBuilder
from .execution.live import LiveExecution
from .logging_setup import get_logger
from .models import Candle, Order, OrderType, Position, Side, SignalType
from .risk.exposure import build_currency_map
from .risk.manager import RiskManager
from .strategy.base import StrategyBase, StrategyContext

log = get_logger(__name__)


class LiveTradingEngine:
    def __init__(
        self,
        strategy: StrategyBase,
        config: TradingConfig,
        rest_client: CapitalRestClient,
        *,
        max_history: int = 1000,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.rest = rest_client
        self.execution = LiveExecution(rest_client)
        self.ws = CapitalWebSocketClient(rest_client)
        self.max_history = max_history

        vpp = config.instruments[0].value_per_point if config.instruments else 1.0
        self.risk = RiskManager(
            config.risk,
            value_per_point=vpp,
            currency_map=build_currency_map(config.instruments),
        )
        self._builders: dict[str, CandleBuilder] = {
            inst.epic: CandleBuilder(inst.epic, inst.timeframe) for inst in config.instruments
        }
        self._history: dict[str, list[Candle]] = {e: [] for e in self._builders}
        self._positions: dict[str, Position] = {}
        self._equity = config.starting_equity

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Authenticate, reconcile positions, warm up history, and stream."""
        self.rest.ensure_session()
        self._reconcile_positions()
        self._warmup_history()
        epics = list(self._builders.keys())
        self.ws.subscribe(epics, self._on_price)
        log.info("live engine starting", extra={"epics": ",".join(epics),
                                                "strategy": self.strategy.name})
        self.ws.run_forever()

    def stop(self) -> None:
        self.ws.stop()

    # ------------------------------------------------------------------ #
    def _reconcile_positions(self) -> None:
        for pos in self.rest.get_positions():
            if pos.epic in self._builders:
                self._positions[pos.epic] = pos
                log.info("reconciled position", extra={"epic": pos.epic,
                                                       "side": pos.side.value, "size": pos.size})

    def _warmup_history(self) -> None:
        for inst in self.config.instruments:
            try:
                bars = self.rest.get_historical_prices(
                    inst.epic, inst.timeframe, max_bars=self.max_history
                )
                self._history[inst.epic] = bars[-self.max_history:]
                log.info("warmed history", extra={"epic": inst.epic, "bars": len(bars)})
            except Exception as exc:
                log.warning("history warmup failed", extra={"epic": inst.epic, "error": str(exc)})

    # ------------------------------------------------------------------ #
    def _on_price(self, update: dict) -> None:
        epic = update.get("epic")
        mid = update.get("mid")
        if epic not in self._builders or mid is None:
            return
        closed = self._builders[epic].update(float(mid))
        if closed is not None:
            self._on_candle(closed)

    def _on_candle(self, candle: Candle) -> None:
        buf = self._history.setdefault(candle.epic, [])
        buf.append(candle)
        if len(buf) > self.max_history:
            del buf[0 : len(buf) - self.max_history]

        context = StrategyContext(
            history=buf,
            position=self._positions.get(candle.epic),
            equity=self._equity,
            params=self.config.strategy_params,
        )
        signal = self.strategy.on_candle(candle, context)
        if signal is None or signal.type is SignalType.HOLD:
            return
        log.info("signal", extra={"epic": candle.epic, "type": signal.type.value})

        if signal.type is SignalType.EXIT:
            self._close(candle.epic)
            return

        existing = self._positions.get(candle.epic)
        if existing is not None:
            if existing.side == signal.side:
                return
            self._close(candle.epic)

        decision = self.risk.evaluate(
            signal, price=candle.close, equity=self._equity,
            positions=list(self._positions.values()),
        )
        if not decision.approved or decision.order is None:
            log.info("signal rejected", extra={"epic": candle.epic, "reason": decision.reason})
            return
        self._open(decision.order, candle.close)

    # ------------------------------------------------------------------ #
    def _open(self, order: Order, price: float) -> None:
        fill = self.execution.execute(order, reference_price=price)
        self._positions[order.epic] = Position(
            epic=order.epic, side=order.side, size=order.size, entry_price=fill.price,
            deal_id=fill.deal_id, stop_loss=order.stop_loss, take_profit=order.take_profit,
        )

    def _close(self, epic: str) -> None:
        pos = self._positions.get(epic)
        if pos is None:
            return
        if pos.deal_id:
            try:
                self.rest.close_position(pos.deal_id)
            except Exception as exc:
                log.error("close failed", extra={"epic": epic, "error": str(exc)})
                return
        self._positions.pop(epic, None)
        log.info("closed position", extra={"epic": epic})

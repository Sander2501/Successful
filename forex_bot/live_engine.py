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

from typing import TYPE_CHECKING, Optional

from .api.rest_client import CapitalRestClient
from .api.websocket_client import CapitalWebSocketClient
from .config import TradingConfig
from .data.candle_builder import CandleBuilder
from .execution.live import LiveExecution
from .logging_setup import get_logger
from .models import Candle, Order, OrderType, Position, Side, SignalType, _parse_ts
from .risk.correlation import CorrelationModel
from .risk.exposure import build_currency_map
from .risk.manager import RiskManager
from .strategy.base import StrategyBase, StrategyContext

if TYPE_CHECKING:
    from .state.store import StateStore

log = get_logger(__name__)


class LiveTradingEngine:
    def __init__(
        self,
        strategy: StrategyBase,
        config: TradingConfig,
        rest_client: CapitalRestClient,
        *,
        max_history: int = 1000,
        state_store: Optional["StateStore"] = None,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.rest = rest_client
        self.execution = LiveExecution(rest_client)
        self.ws = CapitalWebSocketClient(rest_client)
        self.max_history = max_history
        self.state = state_store

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
        self._bars_since_corr = 0

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Restore state, authenticate, reconcile positions, warm up, and stream."""
        self._restore_state()
        self.rest.ensure_session()
        self._reconcile_positions()
        self._warmup_history()
        if self.risk.killed:
            log.warning("restored state has kill switch tripped; flattening and halting")
            for ep in list(self._positions):
                self._close(ep)
            return
        epics = list(self._builders.keys())
        self.ws.subscribe(epics, self._on_price)
        log.info("live engine starting", extra={"epics": ",".join(epics),
                                                "strategy": self.strategy.name})
        self.ws.run_forever()

    def stop(self) -> None:
        self.ws.stop()

    # ------------------------------------------------------------------ #
    def _restore_state(self) -> None:
        """Load persisted risk state (high-water mark, kill flag) on startup."""
        if self.state is None:
            return
        self.risk.restore(self.state.load_risk_state())
        # Persisted positions carry protective levels / deal ids; the broker is
        # the source of truth for what is actually open, so we merge below.
        self._persisted_positions = {p.epic: p for p in self.state.load_positions()}
        if self.risk.killed:
            log.warning("restored kill-switch state: trading is halted")

    def _persist(self) -> None:
        if self.state is None:
            return
        self.state.save_positions(list(self._positions.values()))
        self.state.save_risk_state(self.risk.snapshot())

    def _reconcile_positions(self) -> None:
        persisted = getattr(self, "_persisted_positions", {})
        for pos in self.rest.get_positions():
            if pos.epic in self._builders:
                # Recover protective levels / deal id from persisted state when
                # the broker snapshot omits them.
                prev = persisted.get(pos.epic)
                if prev is not None:
                    pos.stop_loss = pos.stop_loss if pos.stop_loss is not None else prev.stop_loss
                    pos.take_profit = (
                        pos.take_profit if pos.take_profit is not None else prev.take_profit
                    )
                self._positions[pos.epic] = pos
                log.info("reconciled position", extra={"epic": pos.epic,
                                                       "side": pos.side.value, "size": pos.size})
        self._persist()

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

        self._refresh_correlation()

    def _refresh_correlation(self) -> None:
        """(Re)build the correlation model from current rolling history."""
        threshold = self.config.risk.correlation_threshold
        warm = {e: b for e, b in self._history.items() if len(b) >= 3}
        if threshold is not None and len(warm) >= 2:
            self.risk.set_correlation(CorrelationModel.from_candles(warm, threshold))
            self._bars_since_corr = 0

    def _refresh_equity(self, when) -> None:
        """Pull current account equity and drive the daily/kill-switch logic.

        Equity from the broker is the correct source for drawdown limits. On any
        failure we keep the last known value so the limits still function.
        """
        try:
            data = self.rest.get_accounts()
            self._equity = _extract_equity(data, fallback=self._equity)
        except Exception as exc:
            log.warning("equity refresh failed", extra={"error": str(exc)})
        self.risk.update_equity(when.date(), self._equity)

    # ------------------------------------------------------------------ #
    def _on_price(self, update: dict) -> None:
        epic = update.get("epic")
        mid = update.get("mid")
        if epic not in self._builders or mid is None:
            return
        # Bucket by the quote's own (exchange) timestamp when present, not by
        # wall-clock arrival, so candles match the historical bars.
        ts = None
        raw_ts = update.get("timestamp")
        if raw_ts is not None:
            try:
                ts = _parse_ts(raw_ts)
            except Exception:
                ts = None
        closed = self._builders[epic].update(float(mid), ts)
        if closed is not None:
            self._on_candle(closed)

    def _on_candle(self, candle: Candle) -> None:
        buf = self._history.setdefault(candle.epic, [])
        buf.append(candle)
        if len(buf) > self.max_history:
            del buf[0 : len(buf) - self.max_history]

        # Periodically re-estimate correlations (they drift over time).
        refresh = self.config.risk.correlation_refresh_bars
        self._bars_since_corr += 1
        if refresh and self._bars_since_corr >= refresh:
            self._refresh_correlation()

        # Refresh equity and enforce the portfolio kill switch.
        self._refresh_equity(candle.timestamp)
        if self.risk.killed:
            log.warning("kill switch tripped: flattening all positions and stopping")
            for ep in list(self._positions):
                self._close(ep)
            self._persist()
            self.stop()
            return

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
        self._persist()

    def _close(self, epic: str) -> None:
        pos = self._positions.get(epic)
        if pos is None:
            return
        try:
            if pos.deal_id:
                self.rest.close_position(pos.deal_id)
            else:
                self.rest.close_epic(epic)
        except Exception as exc:
            # The stored dealId can be stale/unclosable; fall back to resolving
            # the live position for this epic before giving up.
            log.warning("close by dealId failed; resolving via /positions",
                        extra={"epic": epic, "error": str(exc)})
            try:
                self.rest.close_epic(epic)
            except Exception as exc2:
                log.error("close failed", extra={"epic": epic, "error": str(exc2)})
                return
        self._positions.pop(epic, None)
        if self.state is not None:
            self.state.remove_position(epic)
            self.state.save_risk_state(self.risk.snapshot())
        log.info("closed position", extra={"epic": epic})


def _extract_equity(accounts_payload: dict, *, fallback: float) -> float:
    """Best-effort extraction of account equity from a Capital.com payload.

    Tries the preferred account first, then any account, and several known
    balance fields, so a minor schema variation does not silently disable the
    drawdown limits.
    """
    accounts = accounts_payload.get("accounts") or []
    preferred = next((a for a in accounts if a.get("preferred")), None)
    for acc in ([preferred] if preferred else []) + accounts:
        if not acc:
            continue
        bal = acc.get("balance")
        if isinstance(bal, dict):
            for key in ("balance", "available", "equity"):
                if bal.get(key) is not None:
                    return float(bal[key])
        elif isinstance(bal, (int, float)):
            return float(bal)
    return fallback

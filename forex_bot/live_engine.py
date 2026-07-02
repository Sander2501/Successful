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

from typing import TYPE_CHECKING

from .api.rest_client import CapitalRestClient
from .api.websocket_client import CapitalWebSocketClient
from .config import InstrumentSpecs, TradingConfig
from .data.candle_builder import CandleBuilder
from .execution.live import DealRejectedError, LiveExecution
from .logging_setup import get_logger
from .models import Candle, Order, Position, SignalType, _parse_ts
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
        state_store: StateStore | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.rest = rest_client
        self.execution = LiveExecution(rest_client)
        self.ws = CapitalWebSocketClient(rest_client)
        self.max_history = max_history
        self.state = state_store

        # Per-instrument specs, exactly as the backtester builds them: a basket
        # mixing price scales (EUR/USD ~1.1, USD/JPY ~150) must size each leg
        # with its own value-per-point, not the first instrument's.
        specs = InstrumentSpecs(config.instruments,
                                default_spread=config.costs.spread_points)
        self.specs = specs
        vpp = config.instruments[0].value_per_point if config.instruments else 1.0
        self.risk = RiskManager(
            config.risk,
            value_per_point=vpp,
            currency_map=build_currency_map(config.instruments),
            specs=specs,
        )
        self._builders: dict[str, CandleBuilder] = {
            inst.epic: CandleBuilder(inst.epic, inst.timeframe) for inst in config.instruments
        }
        self._history: dict[str, list[Candle]] = {e: [] for e in self._builders}
        self._positions: dict[str, Position] = {}
        self._equity = config.starting_equity
        self._bars_since_corr = 0
        self._kill_flattened = False
        # Broker minimum deal size per epic (fetched best-effort at warmup) so
        # too-small orders are skipped with a clear log instead of rejected.
        self._min_sizes: dict[str, float] = {}
        # Live bid/ask spread per epic (from the quote stream) for the optional
        # pre-entry spread filter (risk.max_spread_multiple).
        self._last_spread: dict[str, float] = {}
        # Operational halt: after this many consecutive order-execution failures
        # stop opening NEW positions (exits still work) until a restart — a
        # broken execution path must not keep firing orders at the broker.
        self._consec_exec_failures = 0
        self.exec_failure_limit = 5

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
        # Reconcile broker positions after any reconnect: an outage can hide
        # fills/stop-outs, so re-sync local state rather than trust stale memory.
        self.ws.subscribe(epics, self._on_price, on_reconnect=self._on_ws_reconnect)
        log.info("live engine starting", extra={"epics": ",".join(epics),
                                                "strategy": self.strategy.name})
        self.ws.run_forever()

    def _on_ws_reconnect(self) -> None:
        """Re-sync broker state after the price stream drops and recovers."""
        log.info("ws reconnected; reconciling positions")
        try:
            self._reconcile_positions()
        except Exception as exc:
            log.error("post-reconnect reconciliation failed", extra={"error": str(exc)})

    def stop(self) -> None:
        """Stop streaming and persist state, logging what is left open so the
        operator knows exactly what the broker still holds."""
        open_summary = {e: f"{p.side.value} {p.size:g} @ {p.entry_price:g}"
                        for e, p in self._positions.items()}
        log.info("engine stopping",
                 extra={"open_positions": open_summary or "none",
                        "equity": self._equity, "killed": self.risk.killed})
        self._persist()
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
        """Make local positions match the broker (the source of truth).

        Used both at startup and after a reconnect, so it is authoritative: it
        adopts broker positions for our epics and DROPS any local position the
        broker no longer reports (e.g. stopped out during an outage). Protective
        levels missing from the broker snapshot are recovered from the prior
        local position, then from persisted state.
        """
        persisted = getattr(self, "_persisted_positions", {})
        broker = {p.epic: p for p in self.rest.get_positions() if p.epic in self._builders}
        new_positions: dict[str, Position] = {}
        for epic, pos in broker.items():
            prev = self._positions.get(epic) or persisted.get(epic)
            if prev is not None:
                pos.stop_loss = pos.stop_loss if pos.stop_loss is not None else prev.stop_loss
                pos.take_profit = (
                    pos.take_profit if pos.take_profit is not None else prev.take_profit
                )
            new_positions[epic] = pos
            log.info("reconciled position", extra={"epic": epic,
                                                   "side": pos.side.value, "size": pos.size})
        dropped = [e for e in self._positions if e not in new_positions]
        for epic in dropped:
            log.info("position no longer at broker; dropping local", extra={"epic": epic})
        self._positions = new_positions
        self._persist()
        log.info("reconciliation complete",
                 extra={"broker_positions": len(new_positions), "dropped_local": len(dropped),
                        "epics": ",".join(sorted(new_positions)) or "none"})

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

        self._load_dealing_rules()
        self._refresh_correlation()

    def _load_dealing_rules(self) -> None:
        """Fetch each instrument's minimum deal size (best-effort) so orders
        below it are skipped locally instead of fired-and-rejected."""
        for inst in self.config.instruments:
            try:
                details = self.rest.get_market_details(inst.epic)
                value = ((details.get("dealingRules") or {})
                         .get("minDealSize") or {}).get("value")
                if value is not None:
                    self._min_sizes[inst.epic] = float(value)
            except Exception as exc:
                log.warning("could not fetch dealing rules",
                            extra={"epic": inst.epic, "error": str(exc)})

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
        bid, ask = update.get("bid"), update.get("ask")
        if bid is not None and ask is not None:
            self._last_spread[epic] = float(ask) - float(bid)
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
            # Backstop: an exception escaping into the websocket dispatch closes
            # the socket and drops the stream for EVERY epic. One bad candle /
            # broker hiccup must never cost the whole price feed.
            try:
                self._on_candle(closed)
            except Exception:
                log.exception("candle processing failed", extra={"epic": epic})

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
            # Flatten once on the transition into the killed state.
            if not self._kill_flattened:
                log.warning("kill switch tripped: flattening all positions")
                for ep in list(self._positions):
                    self._close(ep)
                self._kill_flattened = True
                self._persist()
            # Recoverable (rolling-window) mode keeps streaming so trading can
            # resume on recovery; all-time-peak mode halts permanently.
            if not self.risk.recoverable:
                self.stop()
            return
        # Recovered: re-arm so a future kill flattens again.
        self._kill_flattened = False

        context = StrategyContext(
            history=buf,
            position=self._positions.get(candle.epic),
            equity=self._equity,
            params=self.config.strategy_params,
        )
        try:
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

            # Operational halt: a broken execution path (repeated failures) must
            # not keep firing new orders at the broker. Exits above still work.
            if self._consec_exec_failures >= self.exec_failure_limit:
                log.error("entry skipped: execution halted after repeated failures "
                          "(restart to clear)",
                          extra={"epic": candle.epic,
                                 "consecutive_failures": self._consec_exec_failures})
                return

            # Optional pre-entry spread filter: skip entries when the live spread
            # is abnormally wide vs the configured per-instrument spread (news,
            # rollover, illiquid hours). Off unless risk.max_spread_multiple set.
            mult = self.config.risk.max_spread_multiple
            base_spread = self.specs.spread(candle.epic)
            live_spread = self._last_spread.get(candle.epic)
            if (mult is not None and live_spread is not None and base_spread > 0
                    and live_spread > mult * base_spread):
                log.info("entry skipped: spread too wide",
                         extra={"epic": candle.epic, "live_spread": live_spread,
                                "limit": mult * base_spread})
                return

            decision = self.risk.evaluate(
                signal, price=candle.close, equity=self._equity,
                positions=list(self._positions.values()),
            )
            if not decision.approved or decision.order is None:
                log.info("signal rejected",
                         extra={"epic": candle.epic, "reason": decision.reason})
                return

            # Broker minimum deal size: skip locally instead of fire-and-reject.
            min_size = self._min_sizes.get(candle.epic)
            if min_size is not None and decision.order.size < min_size:
                log.info("entry skipped: size below broker minimum",
                         extra={"epic": candle.epic, "size": decision.order.size,
                                "min_deal_size": min_size})
                return
            self._open(decision.order, candle.close)
        except DealRejectedError as exc:
            # Expected broker outcome (margin, closed market, size limits): no
            # position exists, nothing recorded locally; try again on a new signal.
            log.warning("entry rejected by broker",
                        extra={"epic": candle.epic, "reason": exc.reason})
        except Exception:
            # Keep the stream alive: a single failed signal/order must not stop
            # candle processing for this or any other instrument.
            log.exception("trade pipeline error", extra={"epic": candle.epic})

    # ------------------------------------------------------------------ #
    def _open(self, order: Order, price: float) -> None:
        try:
            fill = self.execution.execute(order, reference_price=price)
        except Exception:
            self._consec_exec_failures += 1
            if self._consec_exec_failures >= self.exec_failure_limit:
                log.error("execution failure limit reached; halting new entries "
                          "until restart",
                          extra={"consecutive_failures": self._consec_exec_failures,
                                 "limit": self.exec_failure_limit})
            raise
        self._consec_exec_failures = 0
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
                # If the broker reports NO open position for this epic, the local
                # one was stale (rejected entry, or stopped out unseen) — drop it,
                # the broker is the source of truth. Otherwise keep it so a later
                # signal/kill can retry the close.
                try:
                    still_open = self.rest.resolve_position_deal_id(epic, retries=1)
                except Exception:
                    still_open = "unknown"
                if still_open is None:
                    log.warning("no broker position for epic; dropping stale local",
                                extra={"epic": epic})
                else:
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

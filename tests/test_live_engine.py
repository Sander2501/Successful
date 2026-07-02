import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forex_bot.config import InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.live_engine import LiveTradingEngine
from forex_bot.models import Candle, Signal, SignalType
from forex_bot.state.store import StateStore
from forex_bot.strategy.base import StrategyBase
from tests.fakes import FakeRestClient, make_position


class _AlwaysLong(StrategyBase):
    warmup = 2

    def on_candle(self, candle, context):
        if context.position is None and len(context.closes) >= 2:
            return Signal(candle.epic, SignalType.ENTER_LONG, candle.timestamp)
        return None


def _config(max_total_dd=1.0):
    return TradingConfig(
        starting_equity=10000.0,
        instruments=[InstrumentConfig("EURUSD", "MINUTE_15")],
        risk=RiskConfig(max_position_pct=0.5, max_open_positions=2,
                        max_daily_loss_pct=1.0, max_total_drawdown_pct=max_total_dd),
        strategy="ema_crossover",
    )


def _candle(epic, i, price=1.10):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
    return Candle(epic, "MINUTE_15", ts, price, price + 0.0006, price - 0.0006, price, 100.0)


class TestLiveEnginePath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "state.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_warmup_signal_order_and_persist(self):
        fake = FakeRestClient(equity=10000.0)
        store = StateStore(self.db)
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake,
                                   state_store=store, max_history=100)
        engine._restore_state()
        engine._reconcile_positions()
        engine._warmup_history()       # populates history from the fake
        self.assertTrue(engine._history["EURUSD"])

        engine._on_candle(_candle("EURUSD", 100))

        # A position was opened via the broker and persisted to the store.
        self.assertIn("EURUSD", engine._positions)
        self.assertEqual(len(fake.created), 1)
        self.assertEqual(fake.created[0]["direction"], "BUY")
        persisted = {p.epic for p in store.load_positions()}
        self.assertIn("EURUSD", persisted)
        store.close()

    def test_kill_switch_flattens_live_and_persists(self):
        fake = FakeRestClient(equity=10000.0, positions=[make_position("EURUSD")])
        store = StateStore(self.db)
        engine = LiveTradingEngine(_AlwaysLong(), _config(max_total_dd=0.10), fake,
                                   state_store=store, max_history=100)
        engine._restore_state()
        engine._reconcile_positions()  # adopts the broker's open EURUSD position
        engine._warmup_history()
        self.assertIn("EURUSD", engine._positions)

        engine._on_candle(_candle("EURUSD", 100))  # equity 10000 -> peak set
        fake.equity = 8500.0                        # -15% drawdown
        engine._on_candle(_candle("EURUSD", 101))   # triggers the kill switch

        self.assertTrue(engine.risk.killed)
        self.assertEqual(engine._positions, {})            # flattened
        self.assertTrue(fake.closed)                       # broker close called
        self.assertTrue(store.load_risk_state().get("killed"))  # persisted
        store.close()

    def test_idempotent_open_adopts_existing_position(self):
        from forex_bot.execution.live import LiveExecution
        from forex_bot.models import Order, OrderType, Side
        fake = FakeRestClient(equity=10000.0)
        ex = LiveExecution(fake)
        order = Order(epic="EURUSD", side=Side.BUY, size=1000.0,
                      order_type=OrderType.MARKET)
        f1 = ex.execute(order, reference_price=1.10)   # opens
        f2 = ex.execute(order, reference_price=1.10)   # pre-check finds it -> adopt
        self.assertEqual(len(fake.created), 1)         # no duplicate create
        self.assertEqual(f2.deal_id, f1.deal_id)

    def test_rolling_peak_kill_recovers_and_resumes(self):
        fake = FakeRestClient(equity=10000.0)
        cfg = _config(max_total_dd=0.10)
        cfg.risk.drawdown_peak_window_bars = 3
        engine = LiveTradingEngine(_AlwaysLong(), cfg, fake, max_history=100)
        engine._warmup_history()

        engine._on_candle(_candle("EURUSD", 100))   # equity 10000 -> opens long
        self.assertIn("EURUSD", engine._positions)

        fake.equity = 8500.0
        engine._on_candle(_candle("EURUSD", 101))   # -15% -> kill + flatten
        self.assertTrue(engine.risk.killed)
        self.assertEqual(engine._positions, {})

        fake.equity = 10000.0
        engine._on_candle(_candle("EURUSD", 102))   # recovered -> resume + reopen
        self.assertFalse(engine.risk.killed)
        self.assertIn("EURUSD", engine._positions)

    def test_live_risk_uses_per_instrument_specs(self):
        # Sizing must use each instrument's own value-per-point, exactly like the
        # backtester — not the first instrument's for the whole basket.
        cfg = TradingConfig(
            starting_equity=10000.0,
            instruments=[InstrumentConfig("EURUSD", "HOUR_4", value_per_point=1.0),
                         InstrumentConfig("USDJPY", "HOUR_4", value_per_point=100.0)],
            risk=RiskConfig(),
        )
        engine = LiveTradingEngine(_AlwaysLong(), cfg, FakeRestClient(), max_history=10)
        self.assertEqual(engine.risk._vpp("EURUSD"), 1.0)
        self.assertEqual(engine.risk._vpp("USDJPY"), 100.0)

    def test_rejected_deal_records_no_local_position(self):
        # A broker-rejected deal must not create a phantom local position that
        # blocks the epic until the next reconciliation.
        fake = FakeRestClient(equity=10000.0)
        fake.reject_reason = "INSUFFICIENT_FUNDS"
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake, max_history=100)
        engine._warmup_history()

        engine._on_candle(_candle("EURUSD", 100))  # signal fires, broker rejects

        self.assertEqual(len(fake.created), 1)      # the create was attempted
        self.assertNotIn("EURUSD", engine._positions)  # ...but nothing recorded
        # The epic is not blocked: once the broker accepts again, entry works.
        fake.reject_reason = None
        engine._on_candle(_candle("EURUSD", 101))
        self.assertIn("EURUSD", engine._positions)

    def test_close_drops_stale_local_when_broker_has_none(self):
        # A local position the broker no longer reports (rejected entry or an
        # unseen stop-out) must be dropped on close, not stuck forever.
        fake = FakeRestClient(equity=10000.0)
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake, max_history=100)
        engine._positions["EURUSD"] = make_position("EURUSD", deal_id=None)

        engine._close("EURUSD")  # broker: no position -> drop stale local

        self.assertNotIn("EURUSD", engine._positions)

    def test_strategy_exception_does_not_escape_stream_path(self):
        class _Boom(StrategyBase):
            warmup = 0

            def on_candle(self, candle, context):
                raise RuntimeError("strategy bug")

        engine = LiveTradingEngine(_Boom(), _config(), FakeRestClient(), max_history=100)
        engine._warmup_history()
        # Must not raise: an escaping exception would close the websocket and
        # drop the price stream for every instrument.
        engine._on_candle(_candle("EURUSD", 100))
        self.assertNotIn("EURUSD", engine._positions)

    def test_exec_failure_halt_stops_new_entries(self):
        # After N consecutive execution failures, stop firing NEW orders at the
        # broker (a broken path must not spam rejects) until a restart.
        fake = FakeRestClient(equity=10000.0)
        fake.reject_reason = "INSUFFICIENT_FUNDS"
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake, max_history=100)
        engine.exec_failure_limit = 2
        engine._warmup_history()

        engine._on_candle(_candle("EURUSD", 100))   # reject 1
        engine._on_candle(_candle("EURUSD", 101))   # reject 2 -> at the limit
        engine._on_candle(_candle("EURUSD", 102))   # halted: no order attempted

        self.assertEqual(len(fake.created), 2)
        self.assertNotIn("EURUSD", engine._positions)

    def test_exec_failure_counter_resets_on_success(self):
        fake = FakeRestClient(equity=10000.0)
        fake.reject_reason = "INSUFFICIENT_FUNDS"
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake, max_history=100)
        engine.exec_failure_limit = 3
        engine._warmup_history()

        engine._on_candle(_candle("EURUSD", 100))   # reject -> counter 1
        fake.reject_reason = None
        engine._on_candle(_candle("EURUSD", 101))   # success -> counter reset
        self.assertEqual(engine._consec_exec_failures, 0)
        self.assertIn("EURUSD", engine._positions)

    def test_warmup_loads_broker_min_deal_size(self):
        fake = FakeRestClient(equity=10000.0)
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake, max_history=100)
        engine._warmup_history()
        self.assertEqual(engine._min_sizes.get("EURUSD"), 1.0)

    def test_entry_below_min_deal_size_is_skipped(self):
        fake = FakeRestClient(equity=10000.0)
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake, max_history=100)
        engine._warmup_history()
        engine._min_sizes["EURUSD"] = 10_000_000.0  # far above any computed size

        engine._on_candle(_candle("EURUSD", 100))

        self.assertEqual(fake.created, [])           # no order was fired
        self.assertNotIn("EURUSD", engine._positions)

    def test_spread_filter_skips_wide_spread_entries(self):
        cfg = _config()
        cfg.risk.max_spread_multiple = 2.0           # skip when live > 2x configured
        fake = FakeRestClient(equity=10000.0)
        engine = LiveTradingEngine(_AlwaysLong(), cfg, fake, max_history=100)
        engine._warmup_history()

        engine._last_spread["EURUSD"] = 0.0010       # 10x the 0.0001 default
        engine._on_candle(_candle("EURUSD", 100))
        self.assertEqual(fake.created, [])           # skipped, not fired

        engine._last_spread["EURUSD"] = 0.0001       # normal spread -> trades
        engine._on_candle(_candle("EURUSD", 101))
        self.assertIn("EURUSD", engine._positions)

    def test_spread_filter_off_by_default(self):
        fake = FakeRestClient(equity=10000.0)
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake, max_history=100)
        engine._warmup_history()
        engine._last_spread["EURUSD"] = 1.0          # absurd spread, filter off
        engine._on_candle(_candle("EURUSD", 100))
        self.assertIn("EURUSD", engine._positions)

    def test_stop_persists_state(self):
        fake = FakeRestClient(equity=10000.0)
        store = StateStore(self.db)
        engine = LiveTradingEngine(_AlwaysLong(), _config(), fake,
                                   state_store=store, max_history=100)
        engine._warmup_history()
        engine._on_candle(_candle("EURUSD", 100))    # opens a position

        engine.stop()                                 # must persist, not raise

        self.assertIn("EURUSD", {p.epic for p in store.load_positions()})
        self.assertIsNotNone(store.load_risk_state())
        store.close()

    def test_restart_restores_killed_state(self):
        # Session 1: trip and persist the kill switch.
        store = StateStore(self.db)
        store.save_risk_state({"peak_equity": 12000.0, "killed": True,
                               "day": "2026-01-02", "day_start_equity": 12000.0,
                               "halted_for_day": False})
        store.close()

        # Session 2: a fresh engine must come up already halted.
        fake = FakeRestClient(equity=10000.0)
        store2 = StateStore(self.db)
        engine = LiveTradingEngine(_AlwaysLong(), _config(max_total_dd=0.10), fake,
                                   state_store=store2, max_history=100)
        engine._restore_state()
        self.assertTrue(engine.risk.killed)
        store2.close()


if __name__ == "__main__":
    unittest.main()

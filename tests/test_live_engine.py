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

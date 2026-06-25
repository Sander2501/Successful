import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from forex_bot.config import RiskConfig
from forex_bot.models import Position, Side
from forex_bot.risk.manager import RiskManager
from forex_bot.state.store import StateStore


class TestStateStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "state.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_position_round_trip(self):
        positions = [
            Position("EURUSD", Side.BUY, 1000.0, 1.10, deal_id="d1",
                     stop_loss=1.09, take_profit=1.12,
                     opened_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
            Position("GBPUSD", Side.SELL, 500.0, 1.27),
        ]
        with StateStore(self.path) as store:
            store.save_positions(positions)
        with StateStore(self.path) as store:
            loaded = {p.epic: p for p in store.load_positions()}
        self.assertEqual(set(loaded), {"EURUSD", "GBPUSD"})
        eur = loaded["EURUSD"]
        self.assertEqual(eur.side, Side.BUY)
        self.assertAlmostEqual(eur.entry_price, 1.10)
        self.assertEqual(eur.deal_id, "d1")
        self.assertAlmostEqual(eur.stop_loss, 1.09)
        self.assertEqual(eur.opened_at.year, 2026)

    def test_upsert_and_remove(self):
        with StateStore(self.path) as store:
            store.upsert_position(Position("EURUSD", Side.BUY, 1000.0, 1.10))
            store.upsert_position(Position("EURUSD", Side.BUY, 2000.0, 1.11))  # update
            self.assertAlmostEqual(store.load_positions()[0].size, 2000.0)
            store.remove_position("EURUSD")
            self.assertEqual(store.load_positions(), [])

    def test_risk_state_round_trip(self):
        with StateStore(self.path) as store:
            store.save_risk_state({"peak_equity": 11000.0, "killed": True})
        with StateStore(self.path) as store:
            state = store.load_risk_state()
        self.assertEqual(state["peak_equity"], 11000.0)
        self.assertTrue(state["killed"])

    def test_empty_risk_state_default(self):
        with StateStore(self.path) as store:
            self.assertEqual(store.load_risk_state(), {})


class TestRiskSnapshotRestore(unittest.TestCase):
    def test_snapshot_restore_preserves_killswitch(self):
        cfg = RiskConfig(max_total_drawdown_pct=0.15, max_daily_loss_pct=1.0)
        rm = RiskManager(cfg)
        rm.start_day(date(2026, 1, 1), 10000)
        rm.update_equity(date(2026, 1, 1), 12000)  # peak
        rm.update_equity(date(2026, 1, 2), 10000)  # -16.7% -> kill
        self.assertTrue(rm.killed)
        snap = rm.snapshot()

        # Simulate a restart: a fresh manager must remember the kill + peak.
        fresh = RiskManager(cfg)
        self.assertFalse(fresh.killed)
        fresh.restore(snap)
        self.assertTrue(fresh.killed)
        self.assertEqual(fresh._peak_equity, 12000.0)

    def test_restore_empty_is_noop(self):
        rm = RiskManager(RiskConfig())
        rm.restore({})
        self.assertFalse(rm.killed)


if __name__ == "__main__":
    unittest.main()

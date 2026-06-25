import unittest
from datetime import date

from forex_bot.config import RiskConfig
from forex_bot.models import Signal, SignalType
from forex_bot.risk.manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.cfg = RiskConfig(
            risk_per_trade=0.01, max_position_pct=0.5,
            max_open_positions=2, max_daily_loss_pct=0.05,
        )
        self.rm = RiskManager(self.cfg, value_per_point=1.0)

    def test_size_from_stop_distance(self):
        # Equity 10000, risk 1% = 100. Stop distance 0.0050 -> size 20000.
        # But max_position_pct cap = 0.5 * 10000 / price(1.0) = 5000 -> capped.
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.995)
        dec = self.rm.evaluate(sig, price=1.0, equity=10000, open_positions=0)
        self.assertTrue(dec.approved)
        self.assertAlmostEqual(dec.order.size, 5000.0)  # cap applied

    def test_size_uncapped_when_stop_far_enough(self):
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.90)  # distance 0.10
        dec = self.rm.evaluate(sig, price=1.0, equity=10000, open_positions=0)
        # risk 100 / 0.10 = 1000 (< cap 5000)
        self.assertAlmostEqual(dec.order.size, 1000.0)

    def test_rejects_when_max_positions_reached(self):
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.99)
        dec = self.rm.evaluate(sig, price=1.0, equity=10000, open_positions=2)
        self.assertFalse(dec.approved)

    def test_rejects_non_entry(self):
        dec = self.rm.evaluate(Signal("E", SignalType.EXIT), price=1.0, equity=1, open_positions=0)
        self.assertFalse(dec.approved)

    def test_daily_loss_halt(self):
        self.rm.start_day(date(2024, 1, 1), equity=10000)
        self.rm.update_equity(date(2024, 1, 1), equity=9400)  # -6% > 5% limit
        self.assertTrue(self.rm.halted)
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.99)
        dec = self.rm.evaluate(sig, price=1.0, equity=9400, open_positions=0)
        self.assertFalse(dec.approved)


if __name__ == "__main__":
    unittest.main()

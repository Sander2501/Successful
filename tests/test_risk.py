import unittest
from datetime import date

from forex_bot.config import InstrumentConfig, RiskConfig
from forex_bot.models import Position, Side, Signal, SignalType
from forex_bot.risk.exposure import build_currency_map, net_currency_exposures, parse_currencies
from forex_bot.risk.manager import RiskManager


def _pos(epic, side, size=1000.0, price=1.0):
    return Position(epic=epic, side=side, size=size, entry_price=price)


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.cfg = RiskConfig(
            risk_per_trade=0.01, max_position_pct=0.5,
            max_open_positions=2, max_daily_loss_pct=0.05,
        )
        self.rm = RiskManager(self.cfg, value_per_point=1.0)

    def test_size_from_stop_distance(self):
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.995)
        dec = self.rm.evaluate(sig, price=1.0, equity=10000, positions=[])
        self.assertTrue(dec.approved)
        self.assertAlmostEqual(dec.order.size, 5000.0)  # cap applied

    def test_size_uncapped_when_stop_far_enough(self):
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.90)  # distance 0.10
        dec = self.rm.evaluate(sig, price=1.0, equity=10000, positions=[])
        self.assertAlmostEqual(dec.order.size, 1000.0)

    def test_rejects_when_max_positions_reached(self):
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.99)
        held = [_pos("A", Side.BUY), _pos("B", Side.BUY)]
        dec = self.rm.evaluate(sig, price=1.0, equity=10000, positions=held)
        self.assertFalse(dec.approved)

    def test_rejects_non_entry(self):
        dec = self.rm.evaluate(Signal("E", SignalType.EXIT), price=1.0, equity=1, positions=[])
        self.assertFalse(dec.approved)

    def test_daily_loss_halt(self):
        self.rm.start_day(date(2024, 1, 1), equity=10000)
        self.rm.update_equity(date(2024, 1, 1), equity=9400)  # -6% > 5% limit
        self.assertTrue(self.rm.halted)
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.99)
        dec = self.rm.evaluate(sig, price=1.0, equity=9400, positions=[])
        self.assertFalse(dec.approved)


class TestCurrencyExposure(unittest.TestCase):
    def test_parse_currencies(self):
        self.assertEqual(parse_currencies("EURUSD"), ("EUR", "USD"))
        self.assertEqual(parse_currencies("GBP/USD"), ("GBP", "USD"))
        self.assertIsNone(parse_currencies("GOLD"))
        self.assertEqual(parse_currencies("X", ("EUR", "USD")), ("EUR", "USD"))

    def test_net_exposure_stacks_shared_currency(self):
        cmap = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD")}
        positions = [
            Position("EURUSD", Side.BUY, size=1000, entry_price=1.0),
            Position("GBPUSD", Side.BUY, size=1000, entry_price=1.0),
        ]
        net = net_currency_exposures(positions, cmap)
        # Both longs are short USD -> USD exposure stacks to -2000.
        self.assertAlmostEqual(net["USD"], -2000.0)
        self.assertAlmostEqual(net["EUR"], 1000.0)
        self.assertAlmostEqual(net["GBP"], 1000.0)

    def test_currency_cap_blocks_correlated_stack(self):
        cfg = RiskConfig(
            risk_per_trade=1.0, max_position_pct=0.3, max_open_positions=10,
            max_currency_exposure_pct=0.5,
        )
        cmap = build_currency_map([
            InstrumentConfig(epic="EURUSD"), InstrumentConfig(epic="GBPUSD"),
            InstrumentConfig(epic="AUDUSD"),
        ])
        rm = RiskManager(cfg, value_per_point=1.0, currency_map=cmap)
        # Two existing short-USD longs already at 0.3 + 0.3 = 0.6 of equity... but
        # cap is 0.5, so a third USD-quote long must be rejected.
        held = [
            Position("EURUSD", Side.BUY, size=3000, entry_price=1.0),  # USD -3000
            Position("GBPUSD", Side.BUY, size=2000, entry_price=1.0),  # USD -2000
        ]
        sig = Signal("AUDUSD", SignalType.ENTER_LONG, stop_loss=0.99)
        dec = rm.evaluate(sig, price=1.0, equity=10000, positions=held)
        self.assertFalse(dec.approved)
        self.assertIn("USD", dec.reason)

    def test_max_positions_per_currency(self):
        cfg = RiskConfig(max_open_positions=10, max_positions_per_currency=2)
        cmap = build_currency_map([
            InstrumentConfig(epic="EURUSD"), InstrumentConfig(epic="GBPUSD"),
            InstrumentConfig(epic="AUDUSD"),
        ])
        rm = RiskManager(cfg, value_per_point=1.0, currency_map=cmap)
        held = [
            Position("EURUSD", Side.BUY, size=100, entry_price=1.0),
            Position("GBPUSD", Side.BUY, size=100, entry_price=1.0),
        ]
        sig = Signal("AUDUSD", SignalType.ENTER_LONG, stop_loss=0.99)
        dec = rm.evaluate(sig, price=1.0, equity=10000, positions=held)
        self.assertFalse(dec.approved)

    def test_unknown_currency_skips_checks(self):
        cfg = RiskConfig(max_currency_exposure_pct=0.01)
        rm = RiskManager(cfg, value_per_point=1.0, currency_map={})  # no map
        sig = Signal("GOLD", SignalType.ENTER_LONG, stop_loss=0.99)
        dec = rm.evaluate(sig, price=1.0, equity=10000, positions=[])
        self.assertTrue(dec.approved)  # currency checks skipped when unmapped


if __name__ == "__main__":
    unittest.main()

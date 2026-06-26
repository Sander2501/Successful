import unittest

from forex_bot.config import CostConfig, InstrumentConfig, InstrumentSpecs
from forex_bot.execution.simulated import SimulatedExecution
from forex_bot.models import Order, OrderType, Side


class TestInstrumentSpecs(unittest.TestCase):
    def test_lookup_and_fallback(self):
        specs = InstrumentSpecs(
            [InstrumentConfig("EURUSD", value_per_point=1.0, spread_points=0.0001),
             InstrumentConfig("USDJPY", value_per_point=0.9, spread_points=0.02)],
            default_spread=0.0005,
        )
        self.assertAlmostEqual(specs.spread("EURUSD"), 0.0001)
        self.assertAlmostEqual(specs.spread("USDJPY"), 0.02)
        self.assertAlmostEqual(specs.vpp("USDJPY"), 0.9)
        # Unknown epic falls back to defaults.
        self.assertAlmostEqual(specs.spread("AUDUSD"), 0.0005)
        self.assertAlmostEqual(specs.vpp("AUDUSD"), 1.0)

    def test_spread_points_none_uses_default(self):
        specs = InstrumentSpecs([InstrumentConfig("EURUSD")], default_spread=0.0003)
        self.assertAlmostEqual(specs.spread("EURUSD"), 0.0003)


class TestPerInstrumentExecution(unittest.TestCase):
    def test_execution_charges_per_instrument_spread(self):
        specs = InstrumentSpecs(
            [InstrumentConfig("EURUSD", spread_points=0.0001),
             InstrumentConfig("USDJPY", spread_points=0.02)],
            default_spread=0.0001,
        )
        ex = SimulatedExecution(CostConfig(spread_points=0.0001, slippage_points=0.0), specs)
        eur = ex.execute(Order("EURUSD", Side.BUY, 1.0, OrderType.MARKET), reference_price=1.10)
        jpy = ex.execute(Order("USDJPY", Side.BUY, 1.0, OrderType.MARKET), reference_price=150.0)
        # Half-spread applied: EURUSD 0.00005, USDJPY 0.01 — very different.
        self.assertAlmostEqual(eur.price - 1.10, 0.00005)
        self.assertAlmostEqual(jpy.price - 150.0, 0.01)

    def test_without_specs_uses_global_spread(self):
        ex = SimulatedExecution(CostConfig(spread_points=0.0002, slippage_points=0.0))
        fill = ex.execute(Order("ANY", Side.SELL, 1.0, OrderType.MARKET), reference_price=2.0)
        self.assertAlmostEqual(2.0 - fill.price, 0.0001)  # half of 0.0002


if __name__ == "__main__":
    unittest.main()

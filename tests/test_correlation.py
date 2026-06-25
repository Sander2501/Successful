import unittest

from forex_bot.config import InstrumentConfig, RiskConfig
from forex_bot.models import Position, Side, Signal, SignalType
from forex_bot.risk.correlation import CorrelationModel, aligned_returns, pearson
from forex_bot.risk.manager import RiskManager
from tests.helpers import make_candles


def _series_candles(closes, epic):
    return make_candles(closes, epic=epic)


class TestCorrelationMath(unittest.TestCase):
    def test_pearson_perfect_positive(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0]
        self.assertAlmostEqual(pearson(a, b), 1.0, places=6)

    def test_pearson_perfect_negative(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(pearson(a, b), -1.0, places=6)

    def test_aligned_returns_uses_common_timestamps(self):
        # Same length, same timestamps -> aligned return series of equal length.
        a = _series_candles([1.0, 1.1, 1.2, 1.3], "A")
        b = _series_candles([2.0, 1.9, 2.1, 2.2], "B")
        rets = aligned_returns({"A": a, "B": b})
        self.assertEqual(len(rets["A"]), len(rets["B"]))
        self.assertEqual(len(rets["A"]), 3)


class TestCorrelationModel(unittest.TestCase):
    def test_groups_correlated_pairs(self):
        # A and B move together; C moves oppositely-but-correlated; D independent-ish.
        base = [1.0 + 0.01 * i for i in range(60)]
        a = _series_candles(base, "A")
        b = _series_candles([x * 1.0 + 0.0005 for x in base], "B")
        c = _series_candles([3.0 - 0.01 * i for i in range(60)], "C")  # neg corr to A
        model = CorrelationModel.from_candles({"A": a, "B": b, "C": c}, threshold=0.7)
        self.assertTrue(model.has_groups())
        # A and B correlated -> same group.
        self.assertEqual(model.group_of("A"), model.group_of("B"))

    def test_negative_correlation_gets_opposite_sign(self):
        # Build two series with exactly opposite bar-to-bar returns.
        rets = [0.01, -0.02, 0.015, -0.012, 0.008, -0.018, 0.02, -0.01] * 6
        up_closes, down_closes = [1.0], [1.0]
        for r in rets:
            up_closes.append(up_closes[-1] * (1 + r))
            down_closes.append(down_closes[-1] * (1 - r))
        up = _series_candles(up_closes, "UP")
        down = _series_candles(down_closes, "DOWN")
        model = CorrelationModel.from_candles({"UP": up, "DOWN": down}, threshold=0.7)
        # Strongly anti-correlated -> grouped together with opposite signs.
        self.assertEqual(model.group_of("UP"), model.group_of("DOWN"))
        self.assertNotEqual(model.sign("UP"), model.sign("DOWN"))


class TestCorrelationLimits(unittest.TestCase):
    def test_correlated_exposure_cap_blocks_stack(self):
        up = _series_candles([1.0 + 0.01 * i for i in range(60)], "AAA")
        also_up = _series_candles([1.0 + 0.0101 * i for i in range(60)], "BBB")
        model = CorrelationModel.from_candles({"AAA": up, "BBB": also_up}, threshold=0.7)
        self.assertEqual(model.group_of("AAA"), model.group_of("BBB"))

        cfg = RiskConfig(max_open_positions=10, max_position_pct=0.5,
                         max_correlated_exposure_pct=0.6)
        rm = RiskManager(cfg, value_per_point=1.0)
        rm.set_correlation(model)
        # An existing long in the group already at 0.5 of equity; a correlated
        # same-direction long would push the group to 1.0 -> blocked at cap 0.6.
        held = [Position("AAA", Side.BUY, size=5000, entry_price=1.0)]
        sig = Signal("BBB", SignalType.ENTER_LONG, stop_loss=0.99,
                     meta={"atr": 0.005})
        dec = rm.evaluate(sig, price=1.0, equity=10000, positions=held)
        self.assertFalse(dec.approved)
        self.assertIn("correlated", dec.reason)


if __name__ == "__main__":
    unittest.main()

import unittest

from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.research import holdout_test
from tests.helpers import make_candles


def _trending(n=2000):
    import math
    closes, price, regime = [], 1.10, 0.0
    for i in range(n):
        regime = 0.99 * regime + 0.0004 * math.sin(i / 250.0)
        price = max(0.5, price + regime)
        closes.append(price)
    return closes


def _config():
    return TradingConfig(
        starting_equity=10000.0,
        instruments=[InstrumentConfig("TEST", "MINUTE_15")],
        risk=RiskConfig(max_position_pct=0.2, max_open_positions=1),
        costs=CostConfig(spread_points=0.0),
        strategy="ema_crossover",
    )


class TestHoldout(unittest.TestCase):
    def setUp(self):
        self.candles = {"TEST": make_candles(_trending(), epic="TEST")}
        self.config = _config()
        self.grid = {"fast": [5, 10], "slow": [20, 40]}

    def test_holdout_is_disjoint_and_runs(self):
        res = holdout_test(self.candles, self.config, "ema_crossover", self.grid,
                           holdout_frac=0.25, metric="total_return", min_trades=2)
        # The holdout window starts strictly after training ends.
        self.assertGreater(res.holdout_start, res.train_end)
        # Chosen params come from the grid (or empty if nothing met min_trades).
        if res.best_params:
            self.assertIn(res.best_params["fast"], [5, 10])

    def test_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            holdout_test(self.candles, self.config, "ema_crossover", self.grid,
                         holdout_frac=0.8)

    def test_too_little_data_rejected(self):
        small = {"TEST": make_candles(_trending(15), epic="TEST")}
        with self.assertRaises(ValueError):
            holdout_test(small, self.config, "ema_crossover", self.grid, holdout_frac=0.2)


if __name__ == "__main__":
    unittest.main()

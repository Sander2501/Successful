import unittest

from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.research import param_combinations, walk_forward
from forex_bot.research.walkforward import DEFAULT_GRIDS, WalkForwardResult
from forex_bot.strategy import STRATEGY_REGISTRY
from tests.helpers import make_candles


class TestDefaultGridsAndVerdict(unittest.TestCase):
    def test_every_registered_strategy_has_a_grid(self):
        for name in STRATEGY_REGISTRY:
            self.assertIn(name, DEFAULT_GRIDS, f"missing default grid for {name}")

    def test_verdict_flags_edge_and_no_edge(self):
        from forex_bot.cli import _strategy_comparison

        good = WalkForwardResult(strategy="winner", metric="sharpe",
                                 combined_oos_return_pct=3.0, pct_positive_folds=80.0,
                                 combined_profit_factor=1.5, total_oos_trades=40)
        bad = WalkForwardResult(strategy="loser", metric="sharpe",
                                combined_oos_return_pct=-1.0, pct_positive_folds=30.0,
                                combined_profit_factor=0.7, total_oos_trades=40)
        self.assertIn("candidate edge", _strategy_comparison([good, bad]))
        self.assertIn("winner", _strategy_comparison([good, bad]))
        self.assertIn("no strategy showed a robust", _strategy_comparison([bad]))


def _config():
    return TradingConfig(
        starting_equity=10000.0,
        instruments=[InstrumentConfig(epic="TEST", timeframe="MINUTE_15", value_per_point=1.0)],
        risk=RiskConfig(risk_per_trade=0.01, max_position_pct=0.2, max_open_positions=3),
        costs=CostConfig(spread_points=0.0, commission_per_trade=0.0, slippage_points=0.0),
        strategy="ema_crossover",
    )


def _trending_series(n=2600):
    # Deterministic regime-trending series so a trend edge genuinely exists.
    import math

    closes = []
    price = 1.10
    regime = 0.0
    for i in range(n):
        regime = 0.99 * regime + 0.0004 * math.sin(i / 300.0)
        price = max(0.5, price + regime)
        closes.append(price)
    return closes


class TestParamCombinations(unittest.TestCase):
    def test_product(self):
        combos = param_combinations({"a": [1, 2], "b": [3, 4]})
        self.assertEqual(len(combos), 4)
        self.assertIn({"a": 1, "b": 3}, combos)

    def test_empty_grid(self):
        self.assertEqual(param_combinations({}), [{}])


class TestWalkForward(unittest.TestCase):
    def setUp(self):
        self.candles = {"TEST": make_candles(_trending_series(), epic="TEST")}
        self.config = _config()
        self.grid = {"fast": [5, 10], "slow": [20, 40]}

    def test_runs_and_produces_folds(self):
        result = walk_forward(
            self.candles, self.config, "ema_crossover", self.grid,
            is_bars=1200, oos_bars=400, step_bars=400, metric="total_return", min_trades=2,
        )
        self.assertGreater(len(result.folds), 0)
        # Each fold must select params from the grid.
        for fold in result.folds:
            if fold.best_params:
                self.assertIn(fold.best_params["fast"], [5, 10])
                self.assertIn(fold.best_params["slow"], [20, 40])
        # OOS windows must come strictly after their in-sample windows.
        for fold in result.folds:
            self.assertGreaterEqual(fold.oos_start, fold.is_end)

    def test_insufficient_data_raises(self):
        small = {"TEST": make_candles(_trending_series(50), epic="TEST")}
        with self.assertRaises(ValueError):
            walk_forward(small, self.config, "ema_crossover", self.grid,
                         is_bars=1200, oos_bars=400)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            walk_forward(self.candles, self.config, "ema_crossover", self.grid,
                         is_bars=1200, oos_bars=400, metric="nope")


if __name__ == "__main__":
    unittest.main()

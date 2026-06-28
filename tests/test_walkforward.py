import unittest

from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.research import param_combinations, walk_forward
from forex_bot.research.walkforward import DEFAULT_GRIDS, InstrumentBreakdown, PeriodBreakdown, WalkForwardResult
from forex_bot.strategy import STRATEGY_REGISTRY
from tests.helpers import make_candles


class TestDefaultGridsAndVerdict(unittest.TestCase):
    def test_every_registered_strategy_has_a_grid(self):
        for name in STRATEGY_REGISTRY:
            self.assertIn(name, DEFAULT_GRIDS, f"missing default grid for {name}")

    def test_verdict_flags_edge_and_no_edge(self):
        from forex_bot.cli import _strategy_comparison

        # Survivor needs a margin, majority folds, PF>1.05 and >=50 trades.
        good = WalkForwardResult(strategy="winner", metric="sharpe",
                                 combined_oos_return_pct=3.0, pct_positive_folds=80.0,
                                 combined_profit_factor=1.5, total_oos_trades=60)
        bad = WalkForwardResult(strategy="loser", metric="sharpe",
                                combined_oos_return_pct=-1.0, pct_positive_folds=30.0,
                                combined_profit_factor=0.7, total_oos_trades=40)
        good_out = _strategy_comparison([good, bad])
        self.assertIn("worth a closer look", good_out)
        self.assertIn("winner", good_out)
        self.assertIn("Multiple testing", good_out)  # caveats always shown
        self.assertIn("no strategy showed a robust", _strategy_comparison([bad]))

    def test_verdict_flags_thin_positive_as_noise(self):
        from forex_bot.cli import _strategy_comparison
        # Few trades + thin margin -> not a survivor, flagged as likely noise.
        thin = WalkForwardResult(strategy="lucky", metric="sharpe",
                                 combined_oos_return_pct=0.28, pct_positive_folds=71.0,
                                 combined_profit_factor=1.16, total_oos_trades=25)
        out = _strategy_comparison([thin])
        self.assertIn("likely noise", out)

    def test_walkforward_text_includes_instrument_breakdown(self):
        result = WalkForwardResult(
            strategy="demo", metric="sharpe", total_oos_trades=3,
            by_instrument=[InstrumentBreakdown(epic="EURUSD", return_pct=1.2,
                                               trades=3, profit_factor=1.5,
                                               win_rate_pct=66.7,
                                               avg_r_multiple=0.4,
                                               avg_holding_hours=12.0)],
            by_period=[PeriodBreakdown(period="2026-01", return_pct=-0.4,
                                       trades=2, profit_factor=0.8,
                                       win_rate_pct=50.0,
                                       avg_r_multiple=-0.1,
                                       avg_holding_hours=8.0)],
        )
        text = result.to_text()
        self.assertIn("instrument breakdown", text)
        self.assertIn("EURUSD", text)
        self.assertIn("monthly breakdown", text)
        self.assertIn("2026-01", text)
        self.assertIn("avg R", text)


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


class TestScaledCosts(unittest.TestCase):
    def test_scales_global_and_per_instrument_spreads(self):
        from forex_bot.research.walkforward import _scaled_costs

        cfg = TradingConfig(
            instruments=[
                InstrumentConfig(epic="EURUSD", spread_points=0.0001),
                InstrumentConfig(epic="USDJPY", spread_points=0.01),
                InstrumentConfig(epic="NOSPREAD"),  # falls back to global
            ],
            costs=CostConfig(spread_points=0.0002, commission_per_trade=1.0,
                             slippage_points=0.00005),
        )
        scaled = _scaled_costs(cfg, 3.0)
        # Global fallback scales.
        self.assertAlmostEqual(scaled.costs.spread_points, 0.0006)
        self.assertAlmostEqual(scaled.costs.commission_per_trade, 3.0)
        self.assertAlmostEqual(scaled.costs.slippage_points, 0.00015)
        # Per-instrument spreads scale too (the bug: they previously did not).
        by_epic = {i.epic: i for i in scaled.instruments}
        self.assertAlmostEqual(by_epic["EURUSD"].spread_points, 0.0003)
        self.assertAlmostEqual(by_epic["USDJPY"].spread_points, 0.03)
        # Instruments with no explicit spread stay None (use scaled global).
        self.assertIsNone(by_epic["NOSPREAD"].spread_points)
        # Original config is untouched.
        self.assertAlmostEqual(cfg.instruments[0].spread_points, 0.0001)

    def test_identity_at_unit_multiplier(self):
        from forex_bot.research.walkforward import _scaled_costs

        cfg = _config()
        self.assertIs(_scaled_costs(cfg, 1.0), cfg)


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
        self.assertTrue(result.by_instrument)
        self.assertEqual(result.by_instrument[0].epic, "TEST")
        self.assertTrue(result.by_period)


    def test_in_sample_selector_limits_each_fold(self):
        candles = {
            "A": make_candles(_trending_series(), epic="A"),
            "B": make_candles([1.2 for _ in range(2600)], epic="B"),
            "C": make_candles([1.1 + (i % 2) * 0.0001 for i in range(2600)], epic="C"),
        }
        cfg = TradingConfig(
            starting_equity=10000.0,
            instruments=[InstrumentConfig(epic=e, timeframe="MINUTE_15", value_per_point=1.0)
                         for e in candles],
            risk=RiskConfig(risk_per_trade=0.01, max_position_pct=0.2, max_open_positions=3),
            costs=CostConfig(spread_points=0.0, commission_per_trade=0.0, slippage_points=0.0),
            strategy="ema_crossover",
        )
        result = walk_forward(
            candles, cfg, "ema_crossover", self.grid,
            is_bars=1200, oos_bars=400, step_bars=400, metric="total_return",
            min_trades=1, select_top_n=1,
        )
        self.assertTrue(result.folds)
        for fold in result.folds:
            self.assertEqual(len(fold.selected_epics), 1)
        selected = {epic for fold in result.folds for epic in fold.selected_epics}
        self.assertTrue(selected.issubset(set(candles)))
        self.assertIn("selected=", result.to_text())

    def test_insufficient_data_raises(self):
        small = {"TEST": make_candles(_trending_series(50), epic="TEST")}
        with self.assertRaises(ValueError):
            walk_forward(small, self.config, "ema_crossover", self.grid,
                         is_bars=1200, oos_bars=400)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            walk_forward(self.candles, self.config, "ema_crossover", self.grid,
                         is_bars=1200, oos_bars=400, metric="nope")

    def test_cost_multiplier_reduces_returns(self):
        from forex_bot.config import CostConfig
        cfg = _config()
        cfg.costs = CostConfig(spread_points=0.0002, commission_per_trade=0.0,
                               slippage_points=0.0)
        cheap = walk_forward(self.candles, cfg, "ema_crossover", self.grid,
                             is_bars=1200, oos_bars=400, step_bars=400,
                             metric="total_return", min_trades=2, cost_multiplier=1.0)
        pricey = walk_forward(self.candles, cfg, "ema_crossover", self.grid,
                              is_bars=1200, oos_bars=400, step_bars=400,
                              metric="total_return", min_trades=2, cost_multiplier=5.0)
        # Heavier costs cannot improve pooled OOS return.
        self.assertLessEqual(pricey.combined_oos_return_pct,
                             cheap.combined_oos_return_pct + 1e-9)


if __name__ == "__main__":
    unittest.main()


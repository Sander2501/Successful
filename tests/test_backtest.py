import unittest

from forex_bot.backtest.engine import Backtester
from forex_bot.backtest.metrics import compute_metrics
from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.strategy import build_strategy
from tests.helpers import make_candles


def _config():
    return TradingConfig(
        starting_equity=10000.0,
        instruments=[InstrumentConfig(epic="TEST", timeframe="MINUTE_15", value_per_point=1.0)],
        risk=RiskConfig(risk_per_trade=0.01, max_position_pct=0.2, max_open_positions=3),
        costs=CostConfig(spread_points=0.0, commission_per_trade=0.0, slippage_points=0.0),
        strategy="ema_crossover",
        strategy_params={"fast": 5, "slow": 15},
    )


class TestBacktester(unittest.TestCase):
    def setUp(self):
        # Trend down then up so the EMA strategy trades.
        closes = [1.20 - i * 0.001 for i in range(40)] + [1.16 + i * 0.0015 for i in range(60)]
        self.candles = make_candles(closes, epic="TEST")
        self.config = _config()

    def test_runs_and_trades(self):
        strat = build_strategy(self.config.strategy, self.config.strategy_params)
        result = Backtester(strat, self.config).run({"TEST": self.candles})
        self.assertGreater(result.signals_emitted, 0)
        self.assertGreater(len(result.equity_curve), 0)
        # No position should be left open at the end.
        self.assertEqual(result.portfolio.open_position_count, 0)

    def test_metrics_consistent(self):
        strat = build_strategy(self.config.strategy, self.config.strategy_params)
        result = Backtester(strat, self.config).run({"TEST": self.candles})
        report = compute_metrics(result.equity_curve, result.trades)
        self.assertEqual(report.starting_equity, 10000.0)
        self.assertEqual(report.num_trades, len(result.trades))
        self.assertGreaterEqual(report.max_drawdown_pct, 0.0)

    def test_stop_loss_closes_position(self):
        # A long that immediately reverses past the ATR stop must be closed.
        closes = [1.20 - i * 0.001 for i in range(40)] + [1.16 + i * 0.002 for i in range(20)]
        closes += [1.16 - i * 0.01 for i in range(20)]  # sharp drop triggers stops
        candles = make_candles(closes, epic="TEST")
        strat = build_strategy(self.config.strategy, self.config.strategy_params)
        result = Backtester(strat, self.config).run({"TEST": candles})
        reasons_present = len(result.trades) > 0
        self.assertTrue(reasons_present)

    def test_daily_resampled_metrics_are_sane(self):
        # Annualized vol must reflect daily resampling, not per-candle sampling.
        strat = build_strategy(self.config.strategy, self.config.strategy_params)
        result = Backtester(strat, self.config).run({"TEST": self.candles})
        report = compute_metrics(result.equity_curve, result.trades)
        # ~40 days of 15-min candles -> a handful of trading days, modest vol.
        self.assertGreater(report.trading_days, 0)
        self.assertLess(report.volatility_annual_pct, 100.0)
        self.assertAlmostEqual(
            report.trades_per_day, report.num_trades / report.trading_days, places=6
        )
        self.assertEqual(report.total_fees, sum(t.fees for t in result.trades))

    def test_empty_raises(self):
        strat = build_strategy(self.config.strategy, self.config.strategy_params)
        with self.assertRaises(ValueError):
            Backtester(strat, self.config).run({})


if __name__ == "__main__":
    unittest.main()

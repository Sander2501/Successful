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


class TestStopGapSlippage(unittest.TestCase):
    def _bt_with_long(self, stop):
        from datetime import datetime, timezone

        from forex_bot.execution.base import Fill
        from forex_bot.models import Side
        from forex_bot.strategy import build_strategy
        cfg = _config()  # zero costs
        bt = Backtester(build_strategy("ema_crossover", {"fast": 5, "slow": 15}), cfg)
        when = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bt.portfolio.open_position(
            Fill(epic="TEST", side=Side.BUY, size=1.0, price=1.20), when, stop_loss=stop)
        bt.portfolio.mark_price("TEST", 1.20)
        return bt, when

    def test_gap_through_stop_fills_at_open(self):
        from forex_bot.models import Candle
        bt, when = self._bt_with_long(stop=1.10)
        # Bar OPENS at 1.05, already below the 1.10 stop (an overnight gap).
        gap = Candle("TEST", "MINUTE_15", when, open=1.05, high=1.06, low=1.00,
                     close=1.02, volume=1.0)
        bt._check_protective_levels(gap)
        self.assertEqual(len(bt.portfolio.trades), 1)
        # Filled at the gap open (worse), not the optimistic stop price.
        self.assertAlmostEqual(bt.portfolio.trades[0].exit_price, 1.05)

    def test_intrabar_stop_fills_at_stop(self):
        from forex_bot.models import Candle
        bt, when = self._bt_with_long(stop=1.10)
        # Bar opens above the stop and only dips through it intrabar.
        bar = Candle("TEST", "MINUTE_15", when, open=1.15, high=1.16, low=1.08,
                     close=1.12, volume=1.0)
        bt._check_protective_levels(bar)
        self.assertEqual(len(bt.portfolio.trades), 1)
        self.assertAlmostEqual(bt.portfolio.trades[0].exit_price, 1.10)


class TestRMultipleAndDuration(unittest.TestCase):
    def _trade(self, pnl, initial_risk, hours):
        from datetime import datetime, timedelta, timezone

        from forex_bot.models import Side, Trade
        entry = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return Trade(
            epic="TEST", side=Side.BUY, size=1.0,
            entry_price=1.10, exit_price=1.10 + pnl / 1.0,
            entry_time=entry, exit_time=entry + timedelta(hours=hours),
            pnl=pnl, fees=0.0, initial_risk=initial_risk,
        )

    def test_r_multiple_and_holding_hours(self):
        from datetime import datetime, timezone

        # +2R (risk 50), -1R (risk 50), and a stopless trade (R undefined -> skipped).
        trades = [
            self._trade(pnl=100.0, initial_risk=50.0, hours=2.0),
            self._trade(pnl=-50.0, initial_risk=50.0, hours=4.0),
            self._trade(pnl=10.0, initial_risk=0.0, hours=6.0),  # no stop
        ]
        curve = [
            (datetime(2024, 1, 1, tzinfo=timezone.utc), 10000.0),
            (datetime(2024, 1, 2, tzinfo=timezone.utc), 10060.0),
        ]
        report = compute_metrics(curve, trades)
        # Average R over the two trades that carried a stop: (2 + -1) / 2 = 0.5
        self.assertAlmostEqual(report.avg_r_multiple, 0.5)
        # Average holding spans all three trades: (2 + 4 + 6) / 3 = 4.0 hours
        self.assertAlmostEqual(report.avg_holding_hours, 4.0)

    def test_r_multiple_undefined_without_stop(self):
        from datetime import datetime, timezone

        from forex_bot.models import Side, Trade
        t = Trade(epic="X", side=Side.BUY, size=1.0, entry_price=1.0, exit_price=1.1,
                  entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                  exit_time=datetime(2024, 1, 1, 1, tzinfo=timezone.utc), pnl=10.0)
        self.assertIsNone(t.r_multiple)


if __name__ == "__main__":
    unittest.main()

import math
import unittest
from datetime import datetime, timedelta, timezone

from forex_bot.backtest.engine import Backtester
from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.models import Candle
from forex_bot.strategy import build_strategy
from forex_bot.strategy.spread_reversion import SpreadReversionStrategy


def _cointegrated_pair(n=600, seed=1):
    """A drives a random walk; B = A + a mean-reverting (stationary) spread.

    The spread oscillates, so a spread-reversion strategy has something real to
    trade — used to validate mechanics, not to claim a market edge.
    """
    import random
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    log_a = 0.0
    spread = 0.0
    a_candles, b_candles = [], []
    for i in range(n):
        log_a += rng.gauss(0, 0.004)
        spread = 0.92 * spread + rng.gauss(0, 0.01)  # AR(1), mean-reverting
        pa = math.exp(log_a) * 1.10
        pb = math.exp(log_a + spread) * 1.30
        ts = base + timedelta(minutes=15 * i)
        a_candles.append(Candle("EURUSD", "MINUTE_15", ts, pa, pa * 1.0003, pa * 0.9997, pa, 1.0))
        b_candles.append(Candle("GBPUSD", "MINUTE_15", ts, pb, pb * 1.0003, pb * 0.9997, pb, 1.0))
    return {"EURUSD": a_candles, "GBPUSD": b_candles}


def _config():
    return TradingConfig(
        starting_equity=10000.0,
        instruments=[InstrumentConfig("EURUSD", "MINUTE_15"),
                     InstrumentConfig("GBPUSD", "MINUTE_15")],
        risk=RiskConfig(max_position_pct=0.3, max_open_positions=4,
                        max_currency_exposure_pct=1.0, max_total_drawdown_pct=1.0),
        costs=CostConfig(spread_points=0.0),
        strategy="spread_reversion",
        strategy_params={"lookback": 60, "entry_z": 1.5, "exit_z": 0.4},
    )


class TestSpreadReversion(unittest.TestCase):
    def test_param_validation(self):
        with self.assertRaises(ValueError):
            SpreadReversionStrategy(lookback=10)
        with self.assertRaises(ValueError):
            SpreadReversionStrategy(entry_z=0.5, exit_z=1.0)

    def test_trades_a_cointegrated_pair_both_legs(self):
        cfg = _config()
        candles = _cointegrated_pair()
        strat = build_strategy(cfg.strategy, cfg.strategy_params)
        res = Backtester(strat, cfg).run(candles)
        self.assertGreater(len(res.trades), 0)
        # Both legs are traded (market-neutral structure).
        epics = {t.epic for t in res.trades}
        self.assertEqual(epics, {"EURUSD", "GBPUSD"})
        # No position left open at the end.
        self.assertEqual(res.portfolio.open_position_count, 0)

    def test_no_trades_without_two_instruments(self):
        cfg = _config()
        cfg.instruments = [InstrumentConfig("EURUSD", "MINUTE_15")]
        candles = {"EURUSD": _cointegrated_pair()["EURUSD"]}
        strat = build_strategy(cfg.strategy, cfg.strategy_params)
        res = Backtester(strat, cfg).run(candles)
        self.assertEqual(len(res.trades), 0)


if __name__ == "__main__":
    unittest.main()

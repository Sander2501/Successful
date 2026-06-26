"""Tests for the cross-sectional (relative) momentum basket strategy."""

import unittest
from datetime import datetime, timezone

from forex_bot.backtest.engine import Backtester
from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.models import Position, Side, SignalType
from forex_bot.strategy import build_strategy
from forex_bot.strategy.cross_sectional_momentum import CrossSectionalMomentumStrategy
from forex_bot.strategy.portfolio_base import PortfolioContext
from tests.helpers import make_candles


def _basket():
    # Four instruments with a clean momentum ordering: A > B > C > D.
    series = {
        "A": [1.00 + 0.020 * i for i in range(15)],   # strong up
        "B": [1.00 + 0.010 * i for i in range(15)],   # mild up
        "C": [1.00 + 0.000 * i for i in range(15)],   # flat
        "D": [1.00 - 0.010 * i for i in range(15)],   # down
    }
    return {e: make_candles(v, epic=e) for e, v in series.items()}


class TestCrossSectionalMomentum(unittest.TestCase):
    def test_longs_strongest_shorts_weakest(self):
        candles = _basket()
        latest = {e: c[-1] for e, c in candles.items()}
        ctx = PortfolioContext(histories=candles, positions={}, equity=10000.0)
        strat = CrossSectionalMomentumStrategy(lookback=10, top_k=1, rebalance_bars=1)
        sigs = {s.epic: s.type for s in strat.on_bar(latest["A"].timestamp, latest, ctx)}
        self.assertEqual(sigs.get("A"), SignalType.ENTER_LONG)   # strongest
        self.assertEqual(sigs.get("D"), SignalType.ENTER_SHORT)  # weakest

    def test_exits_names_that_leave_the_sleeves(self):
        candles = _basket()
        latest = {e: c[-1] for e, c in candles.items()}
        # Hold C, which momentum will not rank into the top/bottom sleeve.
        positions = {"C": Position(epic="C", side=Side.BUY, size=1.0, entry_price=1.0)}
        ctx = PortfolioContext(histories=candles, positions=positions, equity=10000.0)
        strat = CrossSectionalMomentumStrategy(lookback=10, top_k=1, rebalance_bars=1)
        sigs = {s.epic: s.type for s in strat.on_bar(latest["A"].timestamp, latest, ctx)}
        self.assertEqual(sigs.get("C"), SignalType.EXIT)

    def test_holds_between_rebalances(self):
        candles = _basket()
        latest = {e: c[-1] for e, c in candles.items()}
        ctx = PortfolioContext(histories=candles, positions={}, equity=10000.0)
        strat = CrossSectionalMomentumStrategy(lookback=10, top_k=1, rebalance_bars=5)
        # bars 1..4 are non-rebalance -> no signals; bar 5 rebalances.
        for _ in range(4):
            self.assertEqual(strat.on_bar(latest["A"].timestamp, latest, ctx), [])
        self.assertTrue(strat.on_bar(latest["A"].timestamp, latest, ctx))

    def test_needs_enough_instruments(self):
        candles = _basket()
        sub = {"A": candles["A"]}  # only one instrument: nothing to rank
        latest = {"A": sub["A"][-1]}
        ctx = PortfolioContext(histories=sub, positions={}, equity=10000.0)
        strat = CrossSectionalMomentumStrategy(lookback=10, top_k=1, rebalance_bars=1)
        self.assertEqual(strat.on_bar(latest["A"].timestamp, latest, ctx), [])

    def test_backtest_portfolio_loop_runs_and_trades(self):
        candles = _basket()
        cfg = TradingConfig(
            starting_equity=10000.0,
            instruments=[InstrumentConfig(e) for e in candles],
            risk=RiskConfig(max_position_pct=0.2, max_open_positions=4,
                            max_daily_loss_pct=1.0),
            costs=CostConfig(spread_points=0.0),
            strategy="xsec_momentum",
        )
        strat = build_strategy("xsec_momentum",
                               {"lookback": 10, "top_k": 1, "rebalance_bars": 2})
        result = Backtester(strat, cfg).run(candles)
        self.assertGreaterEqual(result.signals_emitted, 1)
        self.assertGreaterEqual(len(result.trades) + result.portfolio.open_position_count, 1)

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            CrossSectionalMomentumStrategy(lookback=1)
        with self.assertRaises(ValueError):
            CrossSectionalMomentumStrategy(top_k=0)


if __name__ == "__main__":
    unittest.main()

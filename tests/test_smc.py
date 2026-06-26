"""Tests for the SMC sweep-reversal analyzer and strategy.

The scenario is a hand-built liquidity sweep -> CHOCH -> FVG retrace, so the
mechanical detector has an exact, known-good setup to fire on (and a mirror for
the long side), plus negative cases that must NOT fire.
"""

import unittest
from datetime import datetime, timedelta, timezone

from forex_bot.backtest.engine import Backtester
from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.models import Candle
from forex_bot.strategy import build_strategy
from forex_bot.strategy.base import StrategyContext
from forex_bot.strategy.smc import SmcMarketAnalyzer, _confirmed_swings
from tests.helpers import make_candles

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _bar(i, o, h, l, c):
    h = max(h, o, c)
    l = min(l, o, c)
    return Candle(epic="T", timeframe="MINUTE_15",
                  timestamp=START + timedelta(minutes=15 * i),
                  open=o, high=h, low=l, close=c, volume=100.0)


def _short_scenario():
    """A clean short setup: sweep of a swing high, CHOCH below the protected
    swing low, a bearish FVG, and a final candle retracing into the FVG."""
    bars = []
    i = 0
    base = [100, 101, 100.5, 101.5, 100.8, 101.8, 101.0, 102.0, 101.2, 102.2,
            101.4, 102.4, 101.6, 102.6, 101.8, 102.8, 102.0, 103.0, 102.2, 103.2]
    prev = 100.0
    for c in base:
        o = prev
        bars.append(_bar(i, o, max(o, c) + 0.2, min(o, c) - 0.2, c)); prev = c; i += 1
    seq = [
        (103.2, 103.5, 102.5, 102.6),  # 20
        (102.6, 102.8, 101.3, 101.5),  # 21 protected swing low (101.3)
        (101.5, 102.3, 101.4, 102.1),  # 22
        (102.1, 103.1, 102.0, 102.9),  # 23 confirms swing low at 21
        (102.9, 105.0, 102.8, 104.8),  # 24 swing high (105.0) = liquidity
        (104.8, 104.6, 104.0, 104.2),  # 25
        (104.2, 104.5, 103.5, 103.8),  # 26 confirms swing high at 24
        (103.8, 104.2, 103.4, 103.9),  # 27
        (103.9, 105.6, 103.8, 104.3),  # 28 SWEEP above 105.0, closes back below
        (104.3, 104.4, 103.0, 103.1),  # 29
        (103.1, 103.2, 101.5, 101.6),  # 30
        (101.6, 102.0, 100.2, 100.4),  # 31 displacement low + CHOCH (<101.3)
        (100.4, 103.5, 100.3, 103.3),  # 32 retrace into sweep-side FVG [103.2,103.8]
    ]
    for o, h, l, c in seq:
        bars.append(_bar(i, o, h, l, c)); i += 1
    return bars


def _mirror(bars, m=200.0):
    """Reflect a short scenario into the equivalent long scenario."""
    return [Candle(epic="T", timeframe="MINUTE_15", timestamp=c.timestamp,
                   open=2 * m - c.open, high=2 * m - c.low, low=2 * m - c.high,
                   close=2 * m - c.close, volume=100.0) for c in bars]


def _analyzer():
    return SmcMarketAnalyzer(swing_k=2, lookback=30, htf_factor=4, atr_period=14,
                             stop_buffer_atr=0.1, min_rr=1.5, tp_rr=2.0)


class TestSmcAnalyzer(unittest.TestCase):
    def test_short_setup_fires(self):
        a = _analyzer().analyze(_short_scenario())
        self.assertTrue(a.actionable)
        self.assertEqual(a.direction, "short")
        # Structural stop sits above the sweep wick; target below entry.
        self.assertGreater(a.stop, 105.0)
        self.assertLess(a.target, a.entry_zone[0])
        self.assertEqual(a.reasons["zone"], "fvg")
        self.assertAlmostEqual(a.reasons["swept_level"], 105.0)
        self.assertGreaterEqual(a.reasons["rr"], 1.5)

    def test_long_setup_fires(self):
        a = _analyzer().analyze(_mirror(_short_scenario()))
        self.assertTrue(a.actionable)
        self.assertEqual(a.direction, "long")
        self.assertLess(a.stop, a.entry_zone[1])   # stop below the zone
        self.assertGreater(a.target, a.entry_zone[1])

    def test_no_setup_on_smooth_trend(self):
        flat = make_candles([100 + i * 0.05 for i in range(80)], epic="T")
        self.assertFalse(_analyzer().analyze(flat).actionable)

    def test_insufficient_history_returns_empty(self):
        few = make_candles([100, 101, 102], epic="T")
        self.assertFalse(_analyzer().analyze(few).actionable)

    def test_confirmed_swings_never_use_future_bars(self):
        # No-look-ahead guarantee: a fractal of strength k is only emitted once k
        # bars exist on BOTH sides, so no swing index falls within k of the end.
        bars = _short_scenario()
        highs = [c.high for c in bars]
        lows = [c.low for c in bars]
        sh, sl = _confirmed_swings(highs, lows, 2)
        last_usable = len(bars) - 1 - 2
        for idx, _ in sh + sl:
            self.assertLessEqual(idx, last_usable)

    def test_rejects_when_rr_below_minimum(self):
        # A very high min_rr makes the otherwise-valid setup non-actionable.
        strict = SmcMarketAnalyzer(swing_k=2, lookback=30, atr_period=14, min_rr=99.0)
        self.assertFalse(strict.analyze(_short_scenario()).actionable)

    def test_sweep_quality_gate_rejects_shallow_penetration(self):
        # Require the sweep to poke a huge distance beyond the level -> no setup.
        strict = SmcMarketAnalyzer(swing_k=2, lookback=30, atr_period=14,
                                   min_rr=1.5, min_pen_atr=50.0)
        self.assertFalse(strict.analyze(_short_scenario()).actionable)

    def test_fvg_prefer_changes_entry_zone(self):
        # Sweep-side FVG sits higher than the CHOCH-side FVG (better short entry).
        sweep = SmcMarketAnalyzer(swing_k=2, lookback=30, atr_period=14, min_rr=1.5,
                                  tp_rr=2.0, fvg_prefer="sweep").analyze(_short_scenario())
        choch = SmcMarketAnalyzer(swing_k=2, lookback=30, atr_period=14, min_rr=1.5,
                                  tp_rr=2.0, fvg_prefer="choch").analyze(_short_scenario())
        self.assertTrue(sweep.actionable and choch.actionable)
        self.assertGreater(sweep.entry_zone[1], choch.entry_zone[1])

    def test_invalid_fvg_prefer_raises(self):
        with self.assertRaises(ValueError):
            SmcMarketAnalyzer(swing_k=2, lookback=30, fvg_prefer="nonsense")


class TestSmcSessionFilter(unittest.TestCase):
    def _signal_at_last_bar(self, **kwargs):
        from forex_bot.strategy.smc import SmcSweepReversalStrategy
        bars = _short_scenario()
        strat = SmcSweepReversalStrategy(swing_k=2, lookback=30, atr_period=14,
                                         min_rr=1.5, tp_rr=2.0, **kwargs)
        ctx = StrategyContext(history=bars, position=None, equity=10000.0)
        return strat.on_candle(bars[-1], ctx)

    def test_entry_allowed_in_session(self):
        # Final bar is at 08:00 UTC (bar 32 * 15min); a 7..16 window includes it.
        sig = self._signal_at_last_bar(session_start_hour=7, session_end_hour=16)
        self.assertIsNotNone(sig)

    def test_entry_blocked_outside_session(self):
        # A 9..16 window excludes the 08:00 setup.
        sig = self._signal_at_last_bar(session_start_hour=9, session_end_hour=16)
        self.assertIsNone(sig)


class TestSmcStrategyIntegration(unittest.TestCase):
    def _config(self):
        return TradingConfig(
            starting_equity=10000.0,
            instruments=[InstrumentConfig(epic="T", timeframe="MINUTE_15",
                                          value_per_point=1.0)],
            risk=RiskConfig(risk_per_trade=0.01, max_position_pct=0.5,
                            max_open_positions=1),
            costs=CostConfig(spread_points=0.0, commission_per_trade=0.0,
                             slippage_points=0.0),
            strategy="sweep_reversal",
            strategy_params={"swing_k": 2, "lookback": 30, "atr_period": 14,
                             "min_rr": 1.5},
        )

    def test_backtest_runs_and_opens_structural_trade(self):
        bars = _short_scenario()
        # Extend downward so the short reaches its target and closes in-sample.
        i = len(bars)
        for px in [101.0, 99.0, 97.0, 95.0, 94.0]:
            o = bars[-1].close
            bars.append(_bar(i, o, max(o, px) + 0.1, min(o, px) - 0.1, px)); i += 1
        cfg = self._config()
        strat = build_strategy(cfg.strategy, cfg.strategy_params)
        result = Backtester(strat, cfg).run({"T": bars})
        self.assertGreaterEqual(result.signals_emitted, 1)
        self.assertGreaterEqual(len(result.trades), 1)
        # Structural stop => initial risk recorded => R-multiple is defined.
        t = result.trades[0]
        self.assertGreater(t.initial_risk, 0.0)
        self.assertIsNotNone(t.r_multiple)


if __name__ == "__main__":
    unittest.main()

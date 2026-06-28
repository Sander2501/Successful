import unittest

from forex_bot.models import SignalType
from forex_bot.strategy import build_strategy
from forex_bot.strategy.base import StrategyContext
from forex_bot.strategy.donchian_breakout import (
    DonchianBreakoutStrategy,
    _atr_regime_threshold,
    _quantile,
)
from forex_bot.strategy.ema_crossover import EmaCrossoverStrategy
from tests.helpers import make_candles


def run_strategy(strategy, candles):
    """Replay candles, returning the list of (index, signal) emitted."""
    signals = []
    for i in range(len(candles)):
        ctx = StrategyContext(history=candles[: i + 1], position=None, equity=10000)
        sig = strategy.on_candle(candles[i], ctx)
        if sig is not None:
            signals.append((i, sig))
    return signals


class TestEmaCrossover(unittest.TestCase):
    def test_emits_long_on_upcross(self):
        # Down then up trend should produce at least one long entry.
        closes = [1.0 - i * 0.001 for i in range(40)] + [0.96 + i * 0.002 for i in range(40)]
        candles = make_candles(closes)
        strat = EmaCrossoverStrategy(fast=5, slow=15)
        signals = run_strategy(strat, candles)
        self.assertTrue(any(s.type == SignalType.ENTER_LONG for _, s in signals))

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            EmaCrossoverStrategy(fast=20, slow=10)

    def test_warmup_no_signal(self):
        candles = make_candles([1.0, 1.01, 1.02])
        strat = EmaCrossoverStrategy(fast=5, slow=15)
        self.assertEqual(run_strategy(strat, candles), [])


class TestEmaFilters(unittest.TestCase):
    def test_trend_filter_blocks_counter_trend(self):
        # Persistent downtrend: an up-cross should be suppressed by the regime gate.
        closes = [1.30 - i * 0.0008 for i in range(160)]
        # inject a brief bounce to create an up-cross while still below the 100-EMA
        for i in range(160, 180):
            closes.append(closes[-1] + 0.002)
        candles = make_candles(closes)
        unfiltered = run_strategy(EmaCrossoverStrategy(fast=5, slow=15), candles)
        filtered = run_strategy(
            EmaCrossoverStrategy(fast=5, slow=15, trend_filter=100), candles
        )
        longs_unfiltered = [s for _, s in unfiltered if s.type == SignalType.ENTER_LONG]
        longs_filtered = [s for _, s in filtered if s.type == SignalType.ENTER_LONG]
        # The regime gate should not increase counter-trend longs.
        self.assertLessEqual(len(longs_filtered), len(longs_unfiltered))

    def test_min_separation_reduces_signals(self):
        closes = [1.0 + 0.0005 * ((-1) ** i) for i in range(120)]  # tight chop
        candles = make_candles(closes)
        base = run_strategy(EmaCrossoverStrategy(fast=5, slow=15), candles)
        gated = run_strategy(
            EmaCrossoverStrategy(fast=5, slow=15, min_separation_pct=0.001), candles
        )
        self.assertLessEqual(len(gated), len(base))

    def test_trend_filter_must_exceed_slow(self):
        with self.assertRaises(ValueError):
            EmaCrossoverStrategy(fast=5, slow=15, trend_filter=10)


class TestDonchianBreakout(unittest.TestCase):
    def test_emits_long_on_upside_breakout(self):
        # Flat range, then a clean upside breakout in a trending regime.
        closes = [1.10 + 0.0001 * ((-1) ** i) for i in range(60)]
        closes += [1.10 + 0.002 * i for i in range(1, 40)]  # strong breakout up
        candles = make_candles(closes, epic="TEST")
        strat = DonchianBreakoutStrategy(entry=20, exit=10, adx_period=None)
        signals = run_strategy(strat, candles)
        self.assertTrue(any(s.type == SignalType.ENTER_LONG for _, s in signals))

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            DonchianBreakoutStrategy(entry=10, exit=10)  # exit must be < entry

    def test_adx_gate_blocks_choppy_breakouts(self):
        # Choppy data: breakouts exist but ADX is weak, so the gate suppresses them.
        closes = [1.10 + 0.003 * ((-1) ** i) for i in range(120)]
        candles = make_candles(closes, epic="TEST")
        gated = run_strategy(DonchianBreakoutStrategy(entry=10, exit=5, adx_period=14,
                                                      adx_threshold=30.0), candles)
        ungated = run_strategy(DonchianBreakoutStrategy(entry=10, exit=5, adx_period=None),
                               candles)
        entries_gated = [s for _, s in gated if s.is_entry]
        entries_ungated = [s for _, s in ungated if s.is_entry]
        self.assertLessEqual(len(entries_gated), len(entries_ungated))

    def test_atr_regime_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            DonchianBreakoutStrategy(atr_regime_lookback=1)
        with self.assertRaises(ValueError):
            DonchianBreakoutStrategy(atr_regime_quantile=1.1)

    def test_atr_regime_threshold_excludes_current_bar(self):
        # The 100.0 current ATR must not influence the trailing threshold.
        threshold = _atr_regime_threshold([1.0, 2.0, 100.0], lookback=2, quantile=0.5)
        self.assertEqual(threshold, 1.5)

    def test_atr_regime_increases_warmup(self):
        strat = DonchianBreakoutStrategy(
            entry=20,
            exit=10,
            atr_period=14,
            adx_period=None,
            atr_regime_lookback=100,
        )
        self.assertGreaterEqual(strat.warmup, 116)

    def test_quantile_interpolates(self):
        self.assertEqual(_quantile([1.0, 3.0], 0.25), 1.5)


class TestRegistry(unittest.TestCase):
    def test_build_known(self):
        self.assertIsInstance(build_strategy("ema_crossover"), EmaCrossoverStrategy)

    def test_build_donchian(self):
        from forex_bot.strategy import DonchianBreakoutStrategy as D
        self.assertIsInstance(build_strategy("donchian_breakout"), D)

    def test_build_with_params(self):
        strat = build_strategy("ema_crossover", {"fast": 3, "slow": 8})
        self.assertEqual(strat.fast, 3)
        self.assertEqual(strat.slow, 8)

    def test_build_unknown_raises(self):
        with self.assertRaises(KeyError):
            build_strategy("does_not_exist")


if __name__ == "__main__":
    unittest.main()

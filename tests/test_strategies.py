import unittest

from forex_bot.models import SignalType
from forex_bot.strategy import build_strategy
from forex_bot.strategy.base import StrategyContext
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


class TestRegistry(unittest.TestCase):
    def test_build_known(self):
        self.assertIsInstance(build_strategy("ema_crossover"), EmaCrossoverStrategy)

    def test_build_with_params(self):
        strat = build_strategy("ema_crossover", {"fast": 3, "slow": 8})
        self.assertEqual(strat.fast, 3)
        self.assertEqual(strat.slow, 8)

    def test_build_unknown_raises(self):
        with self.assertRaises(KeyError):
            build_strategy("does_not_exist")


if __name__ == "__main__":
    unittest.main()

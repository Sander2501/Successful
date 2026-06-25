import unittest

from forex_bot.indicators import adx, atr, dmi, donchian, ema, rsi, sma


class TestIndicators(unittest.TestCase):
    def test_sma_basic(self):
        out = sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], 3.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_sma_invalid_period(self):
        with self.assertRaises(ValueError):
            sma([1, 2, 3], 0)

    def test_ema_seed_and_length(self):
        values = [float(i) for i in range(1, 21)]
        out = ema(values, 5)
        self.assertEqual(len(out), len(values))
        self.assertIsNone(out[3])
        self.assertAlmostEqual(out[4], 3.0)  # seed = SMA of first 5 = 3.0
        self.assertIsNotNone(out[-1])

    def test_ema_too_short(self):
        self.assertEqual(ema([1, 2], 5), [None, None])

    def test_rsi_all_gains_is_100(self):
        out = rsi([float(i) for i in range(1, 30)], 14)
        self.assertAlmostEqual(out[-1], 100.0)

    def test_rsi_range(self):
        import math

        values = [10 + math.sin(i / 3.0) for i in range(60)]
        out = rsi(values, 14)
        defined = [v for v in out if v is not None]
        self.assertTrue(defined)
        self.assertTrue(all(0.0 <= v <= 100.0 for v in defined))

    def test_atr_positive(self):
        highs = [float(i) + 1 for i in range(30)]
        lows = [float(i) for i in range(30)]
        closes = [float(i) + 0.5 for i in range(30)]
        out = atr(highs, lows, closes, 14)
        self.assertIsNotNone(out[-1])
        self.assertGreater(out[-1], 0)

    def test_adx_strong_in_clean_trend(self):
        # A clean monotonic uptrend should register high ADX and +DI > -DI.
        n = 80
        highs = [10 + i * 0.5 + 0.2 for i in range(n)]
        lows = [10 + i * 0.5 - 0.2 for i in range(n)]
        closes = [10 + i * 0.5 for i in range(n)]
        plus_di, minus_di, adx_series = dmi(highs, lows, closes, 14)
        self.assertIsNotNone(adx_series[-1])
        self.assertGreater(adx_series[-1], 40.0)  # strong trend
        self.assertGreater(plus_di[-1], minus_di[-1])

    def test_adx_low_in_chop(self):
        # Oscillating prices -> weak trend -> low ADX.
        closes = [10 + (1.0 if i % 2 else -1.0) for i in range(80)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        series = adx(highs, lows, closes, 14)
        self.assertIsNotNone(series[-1])
        self.assertLess(series[-1], 30.0)

    def test_donchian_channel(self):
        highs = [1, 3, 2, 5, 4, 6]
        lows = [0, 1, 1, 2, 2, 3]
        upper, lower = donchian([float(h) for h in highs], [float(l) for l in lows], 3)
        self.assertIsNone(upper[1])
        self.assertEqual(upper[2], 3.0)  # max(1,3,2)
        self.assertEqual(upper[3], 5.0)  # max(3,2,5)
        self.assertEqual(lower[3], 1.0)  # min(1,1,2)
        self.assertEqual(upper[-1], 6.0)


if __name__ == "__main__":
    unittest.main()

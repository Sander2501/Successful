import unittest
from datetime import datetime, timezone

from forex_bot.data.candle_builder import CandleBuilder, timeframe_to_seconds


class TestCandleBuilder(unittest.TestCase):
    def test_timeframe_seconds(self):
        self.assertEqual(timeframe_to_seconds("MINUTE_15"), 900)
        self.assertEqual(timeframe_to_seconds("HOUR_4"), 14400)
        with self.assertRaises(ValueError):
            timeframe_to_seconds("NOPE")

    def test_aggregates_into_bars(self):
        cb = CandleBuilder("E", "MINUTE")
        base = datetime(2024, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        # All within minute 0 -> no close yet.
        self.assertIsNone(cb.update(1.0, base))
        self.assertIsNone(cb.update(1.5, base.replace(second=45)))
        # Crossing into minute 1 closes the first bar.
        closed = cb.update(1.2, datetime(2024, 1, 1, 0, 1, 5, tzinfo=timezone.utc))
        self.assertIsNotNone(closed)
        self.assertAlmostEqual(closed.open, 1.0)
        self.assertAlmostEqual(closed.high, 1.5)
        self.assertAlmostEqual(closed.low, 1.0)
        self.assertAlmostEqual(closed.close, 1.5)
        self.assertEqual(closed.timestamp.second, 0)

    def test_flush(self):
        cb = CandleBuilder("E", "MINUTE")
        cb.update(2.0, datetime(2024, 1, 1, 0, 0, 10, tzinfo=timezone.utc))
        closed = cb.flush()
        self.assertIsNotNone(closed)
        self.assertAlmostEqual(closed.close, 2.0)
        self.assertIsNone(cb.flush())


if __name__ == "__main__":
    unittest.main()

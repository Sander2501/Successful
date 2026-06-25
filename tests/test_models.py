import unittest
from datetime import datetime, timezone

from forex_bot.models import Candle, Position, Side, Signal, SignalType


class TestModels(unittest.TestCase):
    def test_candle_requires_tzaware(self):
        with self.assertRaises(ValueError):
            Candle("E", "MINUTE_15", datetime(2024, 1, 1), 1, 1, 1, 1)

    def test_candle_rejects_inconsistent_ohlc(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            Candle("E", "MINUTE_15", ts, open=1.0, high=0.9, low=0.8, close=0.85)

    def test_candle_from_capital_uses_mid(self):
        payload = {
            "snapshotTimeUTC": "2024-01-01T00:00:00",
            "openPrice": {"bid": 1.0, "ask": 1.2},
            "highPrice": {"bid": 1.3, "ask": 1.5},
            "lowPrice": {"bid": 0.9, "ask": 1.1},
            "closePrice": {"bid": 1.1, "ask": 1.3},
            "lastTradedVolume": 500,
        }
        c = Candle.from_capital("EURUSD", "MINUTE_15", payload)
        self.assertAlmostEqual(c.open, 1.1)
        self.assertAlmostEqual(c.close, 1.2)
        self.assertEqual(c.volume, 500.0)
        self.assertEqual(c.timestamp.tzinfo, timezone.utc)

    def test_side_sign_and_opposite(self):
        self.assertEqual(Side.BUY.sign, 1)
        self.assertEqual(Side.SELL.sign, -1)
        self.assertEqual(Side.BUY.opposite, Side.SELL)

    def test_position_pnl(self):
        pos = Position("E", Side.BUY, size=2.0, entry_price=1.0)
        self.assertAlmostEqual(pos.unrealized_pnl(1.5), 1.0)  # (1.5-1.0)*2
        short = Position("E", Side.SELL, size=2.0, entry_price=1.0)
        self.assertAlmostEqual(short.unrealized_pnl(0.5), 1.0)

    def test_signal_side_mapping(self):
        self.assertEqual(Signal("E", SignalType.ENTER_LONG).side, Side.BUY)
        self.assertEqual(Signal("E", SignalType.ENTER_SHORT).side, Side.SELL)
        self.assertIsNone(Signal("E", SignalType.EXIT).side)


if __name__ == "__main__":
    unittest.main()

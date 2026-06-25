import math
import random
import unittest
from datetime import datetime, timedelta, timezone

from forex_bot.models import Candle
from forex_bot.research.screening import pair_stat, screen_pairs


def _candles(epic, prices):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [Candle(epic, "MINUTE_15", base + timedelta(minutes=15 * i),
                  p, p * 1.0003, p * 0.9997, p, 1.0) for i, p in enumerate(prices)]


def _cointegrated(n=600, seed=1):
    # Shared trend dominates the per-bar move (so the legs co-move strongly, as
    # real cointegrated FX pairs do) with a smaller mean-reverting spread on top.
    rng = random.Random(seed)
    log_a = 0.0
    spread = 0.0
    a, b = [], []
    for _ in range(n):
        log_a += rng.gauss(0, 0.006)
        spread = 0.9 * spread + rng.gauss(0, 0.003)  # mean-reverting, smaller
        a.append(math.exp(log_a) * 1.10)
        b.append(math.exp(log_a + spread) * 1.30)
    return a, b


def _independent_walks(n=600, seed=2):
    rng = random.Random(seed)
    la = lb = 0.0
    a, b = [], []
    for _ in range(n):
        la += rng.gauss(0, 0.004)
        lb += rng.gauss(0, 0.004)
        a.append(math.exp(la) * 1.10)
        b.append(math.exp(lb) * 1.30)
    return a, b


class TestScreening(unittest.TestCase):
    def test_cointegrated_pair_is_mean_reverting(self):
        a, b = _cointegrated()
        s = pair_stat("A", _candles("A", a), "B", _candles("B", b))
        self.assertTrue(s.mean_reverting)
        self.assertTrue(s.tradeable_speed, f"half_life={s.half_life}")
        self.assertLess(s.half_life, 100)

    def test_independent_walks_not_cointegrated(self):
        a, b = _independent_walks()
        s = pair_stat("A", _candles("A", a), "B", _candles("B", b))
        # Spurious regression must be rejected by the ADF cointegration test.
        self.assertFalse(s.cointegrated)
        self.assertFalse(s.tradeable_speed)

    def test_ranking_puts_cointegrated_first(self):
        ca, cb = _cointegrated(seed=5)
        ia, ib = _independent_walks(seed=6)
        candles = {
            "COINT_A": _candles("COINT_A", ca),
            "COINT_B": _candles("COINT_B", cb),
            "RAND_X": _candles("RAND_X", ia),
            "RAND_Y": _candles("RAND_Y", ib),
        }
        ranked = screen_pairs(candles)
        top = ranked[0]
        self.assertEqual({top.epic_a, top.epic_b}, {"COINT_A", "COINT_B"})
        self.assertTrue(top.tradeable_speed)


if __name__ == "__main__":
    unittest.main()

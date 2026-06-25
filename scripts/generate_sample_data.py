#!/usr/bin/env python3
"""Generate synthetic OHLCV candles so the backtester can be exercised without
Capital.com credentials. Writes CSV files into the historical data store using
the same schema the live `download` command produces.

Usage:
    python scripts/generate_sample_data.py --epic EURUSD --bars 5000
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta, timezone

from forex_bot.data.candle_builder import timeframe_to_seconds
from forex_bot.data.storage import CandleStore
from forex_bot.models import Candle


def generate(epic: str, timeframe: str, bars: int, seed: int, start_price: float) -> list[Candle]:
    rng = random.Random(seed)
    step = timeframe_to_seconds(timeframe)
    t = datetime.now(timezone.utc) - timedelta(seconds=step * bars)
    price = start_price
    candles: list[Candle] = []
    for i in range(bars):
        # Trend + mean-reversion + noise so crossovers actually occur.
        drift = 0.00002 * math.sin(i / 120.0)
        shock = rng.gauss(0, 0.0006)
        open_p = price
        close_p = max(0.0001, open_p + drift + shock)
        high_p = max(open_p, close_p) + abs(rng.gauss(0, 0.0003))
        low_p = min(open_p, close_p) - abs(rng.gauss(0, 0.0003))
        candles.append(
            Candle(
                epic=epic, timeframe=timeframe,
                timestamp=t,
                open=round(open_p, 5), high=round(high_p, 5),
                low=round(low_p, 5), close=round(close_p, 5),
                volume=round(rng.uniform(100, 1000), 2),
            )
        )
        price = close_p
        t += timedelta(seconds=step)
    return candles


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epic", default="EURUSD")
    p.add_argument("--timeframe", default="MINUTE_15")
    p.add_argument("--bars", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-price", type=float, default=1.10)
    p.add_argument("--data-dir", default="data/historical")
    args = p.parse_args()

    candles = generate(args.epic, args.timeframe, args.bars, args.seed, args.start_price)
    store = CandleStore(args.data_dir)
    path = store.save(args.epic, args.timeframe, candles)
    print(f"Wrote {len(candles)} candles to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

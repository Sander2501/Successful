import unittest
from datetime import datetime, timedelta, timezone

from forex_bot.api.rest_client import CapitalRestClient
from forex_bot.config import CapitalCredentials
from forex_bot.models import Candle


def _creds():
    return CapitalCredentials(environment="demo", identifier="u", password="p", api_key="k")


def _candles(n):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = []
    price = 1.10
    for i in range(n):
        price += 0.0001
        out.append(Candle("EURUSD", "MINUTE_15", base + timedelta(minutes=15 * i),
                          price, price + 0.0005, price - 0.0005, price, 1.0))
    return out


class TestHistoryPaging(unittest.TestCase):
    def test_pages_past_per_request_cap(self):
        client = CapitalRestClient(_creds())
        pool = _candles(2500)

        def fake_get(epic, resolution="MINUTE_15", *, max_bars=1000,
                     from_time=None, to_time=None):
            # Mimic the API: the most recent `max_bars` bars at or before `to`.
            eligible = [c for c in pool if to_time is None or c.timestamp <= to_time]
            return eligible[-max_bars:]

        client.get_historical_prices = fake_get  # type: ignore[assignment]
        out = client.get_historical_prices_paged("EURUSD", "MINUTE_15",
                                                 total=2500, chunk=1000)
        self.assertEqual(len(out), 2500)
        # Chronological and de-duplicated.
        ts = [c.timestamp for c in out]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(len(ts), len(set(ts)))

    def test_stops_when_history_exhausted(self):
        client = CapitalRestClient(_creds())
        pool = _candles(300)  # fewer than requested

        def fake_get(epic, resolution="MINUTE_15", *, max_bars=1000,
                     from_time=None, to_time=None):
            eligible = [c for c in pool if to_time is None or c.timestamp <= to_time]
            return eligible[-max_bars:]

        client.get_historical_prices = fake_get  # type: ignore[assignment]
        out = client.get_historical_prices_paged("EURUSD", total=2000, chunk=1000)
        self.assertEqual(len(out), 300)


if __name__ == "__main__":
    unittest.main()

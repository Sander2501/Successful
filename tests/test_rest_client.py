import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import forex_bot.api.rest_client as rc
from forex_bot.api.rest_client import CapitalRestClient
from forex_bot.config import CapitalCredentials
from forex_bot.models import Candle


class _FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


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


class TestRateLimitRetry(unittest.TestCase):
    def test_retries_on_429_then_succeeds(self):
        client = CapitalRestClient(_creds(), max_retries=3)
        client._cst = "x"
        client._security_token = "y"  # skip login
        responses = iter([_FakeResp(429), _FakeResp(429), _FakeResp(200, {"ok": True})])
        client._session.request = lambda *a, **k: next(responses)  # type: ignore
        orig_sleep = rc.time.sleep
        rc.time.sleep = lambda *_: None  # don't actually back off in the test
        try:
            out = client._request("GET", "/x")
        finally:
            rc.time.sleep = orig_sleep
        self.assertEqual(out, {"ok": True})

    def test_gives_up_after_max_retries(self):
        from forex_bot.api.rest_client import CapitalApiError
        client = CapitalRestClient(_creds(), max_retries=1)
        client._cst = "x"
        client._security_token = "y"
        client._session.request = lambda *a, **k: _FakeResp(429)  # type: ignore
        orig_sleep = rc.time.sleep
        rc.time.sleep = lambda *_: None
        try:
            with self.assertRaises(CapitalApiError) as ctx:
                client._request("GET", "/x")
        finally:
            rc.time.sleep = orig_sleep
        self.assertEqual(ctx.exception.status, 429)


class TestSessionCache(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "session.json")
            c1 = CapitalRestClient(_creds(), session_cache_path=path)
            c1._cst, c1._security_token = "CST1", "TOK1"
            c1._save_session()

            c2 = CapitalRestClient(_creds(), session_cache_path=path)
            self.assertTrue(c2._load_session())
            self.assertEqual(c2._cst, "CST1")
            self.assertEqual(c2._security_token, "TOK1")

    def test_expired_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "session.json"
            c1 = CapitalRestClient(_creds(), session_cache_path=str(path))
            c1._cst, c1._security_token = "CST1", "TOK1"
            c1._save_session()
            data = json.loads(path.read_text())
            data["ts"] = 0  # ancient
            path.write_text(json.dumps(data))

            c2 = CapitalRestClient(_creds(), session_cache_path=str(path))
            self.assertFalse(c2._load_session())

    def test_different_identifier_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "session.json")
            c1 = CapitalRestClient(_creds(), session_cache_path=path)
            c1._cst, c1._security_token = "CST1", "TOK1"
            c1._save_session()

            other = CapitalCredentials(environment="demo", identifier="someone-else",
                                       password="p", api_key="k")
            c2 = CapitalRestClient(other, session_cache_path=path)
            self.assertFalse(c2._load_session())


if __name__ == "__main__":
    unittest.main()

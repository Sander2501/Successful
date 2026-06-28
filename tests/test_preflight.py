import unittest

from forex_bot.config import CapitalCredentials, InstrumentConfig, TradingConfig
from forex_bot.preflight import all_passed, report_text, run_preflight
from tests.fakes import FakeRestClient


def _creds(environment="demo"):
    return CapitalCredentials(environment=environment, identifier="u", password="p",
                              api_key="k")


def _config():
    return TradingConfig(instruments=[InstrumentConfig("EURUSD", "MINUTE_15"),
                                      InstrumentConfig("GBPUSD", "MINUTE_15")])


class TestPreflight(unittest.TestCase):
    def test_all_checks_pass_read_only(self):
        client = FakeRestClient(equity=12345.0)
        results = run_preflight(_config(), _creds(), client=client)
        names = {r.name for r in results}
        self.assertIn("login", names)
        self.assertIn("account/equity", names)
        self.assertIn("history", names)
        self.assertIn("positions", names)
        self.assertTrue(all_passed(results), report_text(results))
        # Equity parsing surfaced the right number.
        eq = next(r for r in results if r.name == "account/equity")
        self.assertIn("12345", eq.detail)
        # Read-only run must not place any orders.
        self.assertEqual(client.created, [])

    def test_login_failure_short_circuits(self):
        client = FakeRestClient()
        def boom():
            raise RuntimeError("bad api key")
        client.login = boom
        results = run_preflight(_config(), _creds(), client=client)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "login")
        self.assertFalse(results[0].ok)
        self.assertFalse(all_passed(results))

    def test_test_order_round_trip_on_demo(self):
        client = FakeRestClient()
        results = run_preflight(_config(), _creds("demo"), client=client, test_order=True)
        self.assertTrue(all_passed(results), report_text(results))
        self.assertEqual(len(client.created), 1)          # one position opened
        self.assertEqual(len(client.closed), 1)           # and closed again
        # Closed by the authoritative position dealId, not the confirm dealId
        # (this is the bug the real --test-order run surfaced).
        self.assertTrue(client.closed[0].startswith("pos-"))
        self.assertFalse(client.closed[0].startswith("confirm-"))

    def test_test_order_refused_on_live(self):
        client = FakeRestClient(environment="live")
        results = run_preflight(_config(), _creds("live"), client=client, test_order=True)
        order_checks = [r for r in results if r.name == "test order"]
        self.assertTrue(order_checks and not order_checks[0].ok)
        self.assertEqual(client.created, [])  # never placed on live


if __name__ == "__main__":
    unittest.main()

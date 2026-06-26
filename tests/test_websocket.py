"""Tests for the WebSocket client's auto-reconnect loop.

A fake ``websocket`` module is injected so the reconnect/backoff lifecycle can be
exercised without a real socket: each fake connection opens, delivers one quote,
then closes, and the test stops the client after a few connects.
"""

import sys
import types
import unittest

from forex_bot.api.websocket_client import CapitalWebSocketClient
from tests.fakes import FakeRestClient

QUOTE = '{"destination":"quote","payload":{"epic":"EURUSD","bid":1.0,"ofr":1.0}}'


class TestWebSocketReconnect(unittest.TestCase):
    def test_reconnects_with_callback_until_stopped(self):
        client = CapitalWebSocketClient(
            FakeRestClient(), reconnect_delay=0.0, max_reconnect_delay=0.0)
        connects = {"n": 0}
        reconnects = {"n": 0}
        messages = {"n": 0}

        client.subscribe(
            ["EURUSD"],
            lambda u: messages.__setitem__("n", messages["n"] + 1),
            on_reconnect=lambda: reconnects.__setitem__("n", reconnects["n"] + 1),
        )

        class FakeWSApp:
            def __init__(self, url, on_open, on_message, on_error, on_close):
                self.on_open = on_open
                self.on_message = on_message
                self.on_close = on_close

            def run_forever(self):
                connects["n"] += 1
                self.on_open(self)
                self.on_message(self, QUOTE)
                if connects["n"] >= 3:
                    client.stop()       # break the outer reconnect loop
                self.on_close(self, 1006, "drop")

            def send(self, *a, **k):
                pass

            def close(self):
                pass

        sys.modules["websocket"] = types.SimpleNamespace(WebSocketApp=FakeWSApp)
        try:
            client.run_forever()
        finally:
            del sys.modules["websocket"]

        self.assertEqual(connects["n"], 3)
        # on_reconnect fires on re-opens only (not the first connect): 2nd + 3rd.
        self.assertEqual(reconnects["n"], 2)
        self.assertGreaterEqual(messages["n"], 3)

    def test_stop_before_run_does_not_loop(self):
        client = CapitalWebSocketClient(FakeRestClient(), reconnect_delay=0.0)
        client.subscribe(["EURUSD"], lambda u: None)

        opened = {"n": 0}

        class FakeWSApp:
            def __init__(self, url, on_open, on_message, on_error, on_close):
                self.on_close = on_close

            def run_forever(self):
                opened["n"] += 1
                client.stop()
                self.on_close(self, 1000, "bye")

            def send(self, *a, **k):
                pass

            def close(self):
                pass

        sys.modules["websocket"] = types.SimpleNamespace(WebSocketApp=FakeWSApp)
        try:
            client.run_forever()
        finally:
            del sys.modules["websocket"]
        self.assertEqual(opened["n"], 1)  # stopped after the first connection


if __name__ == "__main__":
    unittest.main()

"""Capital.com streaming WebSocket client.

Subscribes to live market quotes for a set of epics and dispatches normalized
price updates to a callback. Authentication reuses the REST session tokens
(CST + X-SECURITY-TOKEN). Capital.com allows up to ~40 instruments per session;
the client enforces that limit.

``websocket-client`` is imported lazily so the pure-stdlib core never depends
on it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

from ..logging_setup import get_logger
from .rest_client import CapitalRestClient

log = get_logger(__name__)

MAX_INSTRUMENTS = 40
PriceCallback = Callable[[dict], None]


class CapitalWebSocketClient:
    def __init__(
        self,
        rest_client: CapitalRestClient,
        *,
        ping_interval: float = 30.0,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        self.rest = rest_client
        self.ws_url = rest_client.creds.ws_url
        self.ping_interval = ping_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self._epics: list[str] = []
        self._on_price: PriceCallback | None = None
        self._on_reconnect: Callable[[], None] | None = None
        self._ws = None
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._running = False
        self._stopped = False
        self._connected_before = False
        self._got_message = False

    # ------------------------------------------------------------------ #
    def subscribe(
        self,
        epics: list[str],
        on_price: PriceCallback,
        *,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        if len(epics) > MAX_INSTRUMENTS:
            raise ValueError(
                f"Capital.com allows at most {MAX_INSTRUMENTS} instruments per "
                f"WebSocket session (requested {len(epics)})"
            )
        self._epics = list(epics)
        self._on_price = on_price
        # Invoked after the socket re-opens following a drop, so the caller can
        # reconcile broker state (positions may have changed during the outage).
        self._on_reconnect = on_reconnect

    def run_forever(self) -> None:
        """Connect and block, auto-reconnecting on drops until ``stop()``.

        Each disconnect (that the operator did not request) reconnects with
        capped exponential backoff. The backoff resets after a connection that
        actually received data, so a brief blip recovers fast while a hard outage
        does not hammer the endpoint.
        """
        import websocket  # lazy import

        self._stopped = False
        delay = self.reconnect_delay

        while not self._stopped:
            self._got_message = False
            # Refresh REST session tokens before each (re)connect; a long outage
            # can outlive the ~10-minute session, and the ws auth reuses them.
            try:
                self.rest.ensure_session()
            except Exception as exc:
                log.error("ws session refresh failed before connect",
                          extra={"error": str(exc)})

            self._running = True
            self._ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._make_on_open(),
                on_message=lambda ws, message: self._handle_message(message),
                on_error=lambda ws, error: log.error("ws error", extra={"error": str(error)}),
                on_close=self._make_on_close(),
            )
            self._ws.run_forever()  # blocks until the socket closes
            self._running = False

            if self._stopped:
                break
            # A connection that received data is healthy; reset the backoff.
            if self._got_message:
                delay = self.reconnect_delay
            log.warning("ws disconnected; reconnecting", extra={"sleep": delay})
            time.sleep(delay)
            delay = min(delay * 2.0, self.max_reconnect_delay)

    def _make_on_open(self):
        def _on_open(ws) -> None:
            log.info("ws connected", extra={"epics": ",".join(self._epics)})
            ws.send(json.dumps({
                "destination": "marketData.subscribe",
                "correlationId": "sub-1",
                "cst": self.rest._cst,
                "securityToken": self.rest._security_token,
                "payload": {"epics": self._epics},
            }))
            self._start_pinger(ws)
            # On a *re*-open (not the first connect) let the caller reconcile.
            if self._connected_before and self._on_reconnect is not None:
                try:
                    self._on_reconnect()
                except Exception as exc:
                    log.error("ws on_reconnect handler failed", extra={"error": str(exc)})
            self._connected_before = True
        return _on_open

    def _make_on_close(self):
        def _on_close(ws, status_code, msg) -> None:
            log.warning("ws closed", extra={"code": status_code})
            self._running = False
        return _on_close

    def run_in_thread(self) -> threading.Thread:
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stopped = True
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def _handle_message(self, message: str) -> None:
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        self._got_message = True  # any well-formed frame marks the link healthy
        if data.get("destination") != "quote":
            return
        payload = data.get("payload") or {}
        bid = payload.get("bid")
        ask = payload.get("ofr", payload.get("ask"))
        update = {
            "epic": payload.get("epic"),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0 if bid is not None and ask is not None else None,
            "timestamp": payload.get("timestamp"),
        }
        if self._on_price is not None and update["epic"]:
            self._on_price(update)

    def _start_pinger(self, ws) -> None:
        def _ping_loop() -> None:
            while self._running:
                time.sleep(self.ping_interval)
                try:
                    ws.send(json.dumps({
                        "destination": "ping",
                        "correlationId": "ping",
                        "cst": self.rest._cst,
                        "securityToken": self.rest._security_token,
                    }))
                except Exception:
                    break

        self._ping_thread = threading.Thread(target=_ping_loop, daemon=True)
        self._ping_thread.start()

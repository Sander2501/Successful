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
from typing import Callable, Optional

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
    ) -> None:
        self.rest = rest_client
        self.ws_url = rest_client.creds.ws_url
        self.ping_interval = ping_interval
        self._epics: list[str] = []
        self._on_price: Optional[PriceCallback] = None
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------ #
    def subscribe(self, epics: list[str], on_price: PriceCallback) -> None:
        if len(epics) > MAX_INSTRUMENTS:
            raise ValueError(
                f"Capital.com allows at most {MAX_INSTRUMENTS} instruments per "
                f"WebSocket session (requested {len(epics)})"
            )
        self._epics = list(epics)
        self._on_price = on_price

    def run_forever(self) -> None:
        """Connect and block, processing messages until stopped."""
        import websocket  # lazy import

        self.rest.ensure_session()
        self._running = True

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

        def _on_message(ws, message: str) -> None:
            self._handle_message(message)

        def _on_error(ws, error) -> None:
            log.error("ws error", extra={"error": str(error)})

        def _on_close(ws, status_code, msg) -> None:
            log.warning("ws closed", extra={"code": status_code})
            self._running = False

        self._ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        self._ws.run_forever()

    def run_in_thread(self) -> threading.Thread:
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
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

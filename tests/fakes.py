"""In-memory fakes that mimic the Capital.com REST client for tests.

Lets the live engine and preflight be validated end-to-end without a broker or
network. Response shapes mirror the documented Capital.com payloads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from forex_bot.models import Candle, Position, Side


class FakeRestClient:
    def __init__(self, *, equity: float = 10000.0, positions=None,
                 history_bars: int = 60, ws_url: str = "wss://fake/connect",
                 environment: str = "demo"):
        self.creds = SimpleNamespace(ws_url=ws_url, environment=environment,
                                     is_live=(environment == "live"))
        self.equity = equity
        self._positions = list(positions or [])
        self.history_bars = history_bars
        self.created: list[dict] = []
        self.closed: list[str] = []
        self._open_deals: dict[str, str] = {}  # epic -> position dealId
        self._cst = "fake-cst"
        self._security_token = "fake-token"
        self._deal_seq = 0
        # When set, the next create is REJECTED at confirm time (no position
        # opens at the broker), mirroring e.g. an insufficient-margin rejection.
        self.reject_reason: str | None = None

    # session ---------------------------------------------------------------
    def login(self):
        return None

    def ensure_session(self):
        return None

    def server_time(self):
        return {"serverTime": 0}

    # market data -----------------------------------------------------------
    def get_accounts(self):
        return {"accounts": [{"accountId": "A1", "preferred": True,
                              "balance": {"balance": self.equity, "available": self.equity}}]}

    def get_market_details(self, epic):
        return {"snapshot": {"marketStatus": "TRADEABLE"},
                "dealingRules": {"minDealSize": {"value": 1.0}}}

    def search_markets(self, term):
        return {"markets": [{"epic": term}]}

    def get_historical_prices(self, epic, resolution="MINUTE_15", *, max_bars=10,
                              from_time=None, to_time=None):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        out = []
        price = 1.10
        for i in range(min(max_bars, self.history_bars)):
            price += 0.0005 if i % 2 else -0.0004
            out.append(Candle(epic, resolution, base + timedelta(minutes=15 * i),
                              price, price + 0.0006, price - 0.0006, price, 100.0))
        return out

    def get_positions(self):
        return list(self._positions)

    # trading ---------------------------------------------------------------
    def create_position(self, epic, direction, size, *, stop_level=None,
                        profit_level=None, guaranteed_stop=False):
        self._deal_seq += 1
        ref = f"ref-{self._deal_seq}"
        self.created.append({"epic": epic, "direction": direction, "size": size,
                             "stop_level": stop_level, "ref": ref})
        if self.reject_reason is None:
            self._open_deals[epic] = f"pos-{self._deal_seq}"  # authoritative dealId
        return {"dealReference": ref}

    def confirm_deal(self, deal_reference):
        if self.reject_reason is not None:
            return {"dealReference": deal_reference, "dealStatus": "REJECTED",
                    "rejectReason": self.reject_reason, "level": None}
        # NB: this dealId intentionally differs from the position dealId, to
        # mirror the real API where the confirm id isn't directly closeable.
        return {"dealReference": deal_reference, "dealId": f"confirm-{deal_reference}",
                "dealStatus": "ACCEPTED", "level": 1.10}

    def resolve_position_deal_id(self, epic, *, deal_reference=None,
                                 retries=4, delay=0.0):
        return self._open_deals.get(epic)

    def close_position(self, deal_id):
        self.closed.append(deal_id)
        # Remove whichever epic maps to this dealId.
        for epic, did in list(self._open_deals.items()):
            if did == deal_id:
                del self._open_deals[epic]
        return {"dealReference": "close", "dealId": deal_id}

    def close_epic(self, epic, *, deal_reference=None):
        deal_id = self.resolve_position_deal_id(epic)
        if deal_id is None:
            raise RuntimeError(f"no open position for {epic}")
        return self.close_position(deal_id)


def make_position(epic="EURUSD", side=Side.BUY, size=1000.0, price=1.10, deal_id="d0"):
    return Position(epic=epic, side=side, size=size, entry_price=price, deal_id=deal_id)

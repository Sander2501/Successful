"""Capital.com REST API client.

Implements session management (CST + X-SECURITY-TOKEN), historical price
retrieval, account/position queries, and order placement against the documented
Open API (https://open-api.capital.com). Tokens are refreshed automatically on
401 responses.

Only ``requests`` is needed at runtime; it is imported lazily so importing this
module never fails in environments that only run the pure-stdlib core.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import CapitalCredentials
from ..logging_setup import get_logger
from ..models import Candle, Position
from .rate_limiter import RateLimiter

log = get_logger(__name__)

# Capital.com resolution codes accepted by the /prices endpoint.
VALID_RESOLUTIONS = {
    "MINUTE", "MINUTE_5", "MINUTE_15", "MINUTE_30",
    "HOUR", "HOUR_4", "DAY", "WEEK",
}


class CapitalApiError(RuntimeError):
    def __init__(self, status: int, message: str, payload: Any = None) -> None:
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.payload = payload


class CapitalRestClient:
    def __init__(
        self,
        credentials: CapitalCredentials,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: float = 15.0,
    ) -> None:
        import requests  # lazy import

        self.creds = credentials
        self.base_url = credentials.base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limiter = rate_limiter or RateLimiter()
        self._session = requests.Session()
        self._cst: Optional[str] = None
        self._security_token: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # session
    # ------------------------------------------------------------------ #
    def login(self) -> None:
        """Create a trading session and capture auth tokens."""
        password = self.creds.api_password or self.creds.password
        resp = self._raw_request(
            "POST",
            "/api/v1/session",
            # encryptedPassword=false uses the plaintext flow (over TLS); the
            # encrypted flow would require GET /session/encryptionKey first.
            json={
                "identifier": self.creds.identifier,
                "password": password,
                "encryptedPassword": False,
            },
            headers={"X-CAP-API-KEY": self.creds.api_key},
            authed=False,
        )
        if resp.status_code >= 400:
            raise CapitalApiError(resp.status_code, _err_message(resp), _safe_json(resp))
        self._cst = resp.headers.get("CST")
        self._security_token = resp.headers.get("X-SECURITY-TOKEN")
        if not self._cst or not self._security_token:
            raise CapitalApiError(resp.status_code, "login did not return session tokens")
        log.info("capital.com session established", extra={"env": self.creds.environment})

    def server_time(self) -> dict[str, Any]:
        """Lightweight unauthenticated-ish health endpoint (used by preflight)."""
        return self._request("GET", "/api/v1/time")

    def ensure_session(self) -> None:
        if not self._cst or not self._security_token:
            self.login()

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "X-CAP-API-KEY": self.creds.api_key,
            "CST": self._cst or "",
            "X-SECURITY-TOKEN": self._security_token or "",
        }

    # ------------------------------------------------------------------ #
    # market data
    # ------------------------------------------------------------------ #
    def get_historical_prices(
        self,
        epic: str,
        resolution: str = "MINUTE_15",
        *,
        max_bars: int = 1000,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> list[Candle]:
        """Fetch OHLC history for ``epic`` and map to :class:`Candle` objects."""
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(f"invalid resolution '{resolution}'; one of {VALID_RESOLUTIONS}")
        params: dict[str, Any] = {"resolution": resolution, "max": max_bars}
        if from_time:
            params["from"] = _iso(from_time)
        if to_time:
            params["to"] = _iso(to_time)
        data = self._request("GET", f"/api/v1/prices/{epic}", params=params)
        prices = data.get("prices", [])
        return [Candle.from_capital(epic, resolution, p) for p in prices]

    def get_market_details(self, epic: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/markets/{epic}")

    def search_markets(self, search_term: str) -> dict[str, Any]:
        return self._request("GET", "/api/v1/markets", params={"searchTerm": search_term})

    # ------------------------------------------------------------------ #
    # account / positions
    # ------------------------------------------------------------------ #
    def get_accounts(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/accounts")

    def get_positions(self) -> list[Position]:
        data = self._request("GET", "/api/v1/positions")
        return [Position.from_capital(p) for p in data.get("positions", [])]

    # ------------------------------------------------------------------ #
    # trading
    # ------------------------------------------------------------------ #
    def create_position(
        self,
        epic: str,
        direction: str,
        size: float,
        *,
        stop_level: Optional[float] = None,
        profit_level: Optional[float] = None,
        guaranteed_stop: bool = False,
    ) -> dict[str, Any]:
        """Open a market position. Returns the dealReference payload."""
        body: dict[str, Any] = {
            "epic": epic,
            "direction": direction.upper(),
            "size": size,
            "guaranteedStop": guaranteed_stop,
        }
        if stop_level is not None:
            body["stopLevel"] = stop_level
        if profit_level is not None:
            body["profitLevel"] = profit_level
        return self._request("POST", "/api/v1/positions", json=body)

    def close_position(self, deal_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/positions/{deal_id}")

    def confirm_deal(self, deal_reference: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/confirms/{deal_reference}")

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        _retry: bool = True,
    ) -> dict[str, Any]:
        self.ensure_session()
        resp = self._raw_request(method, path, params=params, json=json,
                                 headers=self._auth_headers, authed=True)
        if resp.status_code == 401 and _retry:
            log.warning("session expired; re-authenticating")
            with self._lock:
                self._cst = self._security_token = None
            self.login()
            return self._request(method, path, params=params, json=json, _retry=False)
        if resp.status_code >= 400:
            raise CapitalApiError(resp.status_code, _err_message(resp), _safe_json(resp))
        return _safe_json(resp) or {}

    def _raw_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        authed: bool = True,
    ):
        self.rate_limiter.acquire()
        url = f"{self.base_url}{path}"
        all_headers = {"Content-Type": "application/json"}
        if headers:
            all_headers.update(headers)
        return self._session.request(
            method, url, params=params, json=json,
            headers=all_headers, timeout=self.timeout,
        )


# --------------------------------------------------------------------------- #
def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Capital.com expects "yyyy-MM-ddTHH:mm:ss" (UTC, no offset).
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _safe_json(resp) -> Any:
    try:
        return resp.json()
    except Exception:
        return None


def _err_message(resp) -> str:
    payload = _safe_json(resp)
    if isinstance(payload, dict):
        return payload.get("errorCode") or payload.get("error") or resp.text[:200]
    return resp.text[:200] if resp.text else "request failed"

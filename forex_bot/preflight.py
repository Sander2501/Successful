"""Live-path preflight checks against the Capital.com API.

Run this with real demo credentials before any live/demo trading session to
validate the full broker path end-to-end:

    login -> account/equity parse -> market details -> historical candles ->
    open positions -> (optional) place + confirm + close a minimal test order

Each step is reported pass/fail with detail. Everything except the optional
``--test-order`` is read-only. Designed to be injected with a fake client so the
orchestration is unit-testable without a broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .config import CapitalCredentials, TradingConfig
from .live_engine import _extract_equity
from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


def run_preflight(
    config: TradingConfig,
    creds: CapitalCredentials,
    *,
    client: Optional[Any] = None,
    test_order: bool = False,
) -> list[CheckResult]:
    """Execute the preflight checks and return their results."""
    results: list[CheckResult] = []
    if client is None:
        from .api.rest_client import CapitalRestClient
        client = CapitalRestClient(creds)

    # 1. Session login (captures CST + X-SECURITY-TOKEN).
    try:
        client.login()
        results.append(CheckResult("login", True, f"environment={creds.environment}"))
    except Exception as exc:
        results.append(CheckResult("login", False, str(exc)))
        return results  # nothing else can run without a session

    # 2. Account + equity parsing (the source for the drawdown kill switch).
    try:
        data = client.get_accounts()
        equity = _extract_equity(data, fallback=float("nan"))
        ok = equity == equity  # NaN check
        results.append(CheckResult(
            "account/equity", ok,
            f"equity={equity:.2f}" if ok else "could not parse account balance",
        ))
    except Exception as exc:
        results.append(CheckResult("account/equity", False, str(exc)))

    if not config.instruments:
        results.append(CheckResult("instruments", False, "no instruments configured"))
        return results

    first = config.instruments[0]

    # 3. Market details for the first configured epic.
    try:
        details = client.get_market_details(first.epic)
        snapshot = details.get("snapshot", {}) if isinstance(details, dict) else {}
        status = snapshot.get("marketStatus", "?")
        results.append(CheckResult("market details", True,
                                   f"{first.epic} status={status}"))
    except Exception as exc:
        results.append(CheckResult("market details", False, f"{first.epic}: {exc}"))

    # 4. Historical candles for every configured instrument.
    for inst in config.instruments:
        try:
            bars = client.get_historical_prices(inst.epic, inst.timeframe, max_bars=10)
            ok = len(bars) > 0
            detail = f"{inst.epic} {inst.timeframe}: {len(bars)} bars"
            if ok:
                detail += f", last close={bars[-1].close:.5f}"
            results.append(CheckResult("history", ok, detail))
        except Exception as exc:
            results.append(CheckResult("history", False, f"{inst.epic}: {exc}"))

    # 5. Open positions listing (reconciliation source).
    try:
        positions = client.get_positions()
        results.append(CheckResult("positions", True, f"{len(positions)} open"))
    except Exception as exc:
        results.append(CheckResult("positions", False, str(exc)))

    # 6. Optional round-trip test order (demo only): open a minimal position and
    #    immediately close it, confirming order placement and reconciliation.
    if test_order:
        results.extend(_test_order(client, first.epic, creds))

    return results


def _test_order(client: Any, epic: str, creds: CapitalCredentials) -> list[CheckResult]:
    if creds.is_live:
        return [CheckResult("test order", False,
                            "refusing to place a test order on a LIVE account")]
    out: list[CheckResult] = []
    try:
        details = client.get_market_details(epic)
        rules = (details or {}).get("dealingRules", {})
        min_size = rules.get("minDealSize", {}).get("value", 1.0)
    except Exception as exc:
        return [CheckResult("test order", False, f"could not read dealing rules: {exc}")]

    try:
        resp = client.create_position(epic, "BUY", float(min_size))
        ref = resp.get("dealReference")
        confirm = client.confirm_deal(ref) if ref else {}
        deal_id = confirm.get("dealId")
        status = confirm.get("dealStatus") or confirm.get("status")
        out.append(CheckResult("test order open", deal_id is not None,
                               f"size={min_size} status={status} dealId={deal_id}"))
        if deal_id:
            client.close_position(deal_id)
            out.append(CheckResult("test order close", True, f"closed dealId={deal_id}"))
    except Exception as exc:
        out.append(CheckResult("test order", False, str(exc)))
    return out


def report_text(results: list[CheckResult]) -> str:
    passed = sum(1 for r in results if r.ok)
    body = "\n".join("  " + r.line() for r in results)
    return f"Preflight: {passed}/{len(results)} checks passed\n{body}"


def all_passed(results: list[CheckResult]) -> bool:
    return bool(results) and all(r.ok for r in results)

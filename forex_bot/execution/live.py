"""Live execution against Capital.com via the REST client.

Translates approved Orders into ``create_position`` calls and confirms the
resulting deal. Kept deliberately thin; reconciliation of broker state lives in
the live engine.
"""

from __future__ import annotations

from ..api.rest_client import CapitalRestClient
from ..logging_setup import get_logger
from ..models import Order
from .base import ExecutionEngine, Fill

log = get_logger(__name__)


class LiveExecution(ExecutionEngine):
    def __init__(self, client: CapitalRestClient) -> None:
        self.client = client

    def execute(self, order: Order, *, reference_price: float) -> Fill:
        # Idempotency guard. Capital.com's create-position has no client-supplied
        # order id, so a create whose response was lost (network blip) cannot be
        # de-duplicated by the broker. Before opening, reconcile against live
        # positions: if one already exists for this epic, a prior create
        # succeeded — adopt it instead of opening a duplicate. A single,
        # no-retry lookup keeps this cheap on the hot path.
        try:
            existing = self.client.resolve_position_deal_id(order.epic, retries=1)
        except Exception as exc:
            existing = None
            log.warning("idempotency pre-check failed; proceeding to open",
                        extra={"epic": order.epic, "error": str(exc)})
        if existing is not None:
            log.warning("idempotency: open position already exists; skipping duplicate",
                        extra={"epic": order.epic, "deal_id": existing})
            return Fill(epic=order.epic, side=order.side, size=order.size,
                        price=reference_price, deal_id=existing)

        resp = self.client.create_position(
            epic=order.epic,
            direction=order.side.value,
            size=order.size,
            stop_level=order.stop_loss,
            profit_level=order.take_profit,
        )
        deal_ref = resp.get("dealReference")
        fill_price = reference_price
        deal_id = None
        if deal_ref:
            try:
                confirm = self.client.confirm_deal(deal_ref)
                fill_price = float(confirm.get("level", reference_price))
            except Exception as exc:  # confirmation is best-effort
                log.warning("deal confirmation failed", extra={"ref": deal_ref, "error": str(exc)})
            # The confirm dealId is not reliably closeable; resolve the
            # authoritative position dealId from /positions so later closes work.
            try:
                deal_id = self.client.resolve_position_deal_id(
                    order.epic, deal_reference=deal_ref
                )
            except Exception as exc:
                log.warning("could not resolve position dealId",
                            extra={"epic": order.epic, "error": str(exc)})
        log.info("live order placed", extra={"epic": order.epic, "side": order.side.value,
                                             "size": order.size, "ref": deal_ref,
                                             "deal_id": deal_id})
        return Fill(
            epic=order.epic,
            side=order.side,
            size=order.size,
            price=fill_price,
            deal_id=deal_id,
        )

"""Core domain models.

Implemented with stdlib dataclasses (no pydantic dependency) so the strategy
and backtesting layers stay import-light and trivially testable. Mapping
helpers convert raw Capital.com API payloads into these typed models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    """Direction of an order or position."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL — handy for PnL math."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class SignalType(str, Enum):
    """The intent emitted by a strategy. Strategies never touch the broker."""

    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"  # close any open position for the instrument
    HOLD = "HOLD"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar for one instrument and timeframe.

    `timestamp` is the bar's *open* time and must be timezone-aware (UTC).
    """

    epic: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Candle.timestamp must be timezone-aware (UTC)")
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(
                f"Inconsistent OHLC for {self.epic} @ {self.timestamp.isoformat()}: "
                f"O={self.open} H={self.high} L={self.low} C={self.close}"
            )

    @classmethod
    def from_capital(cls, epic: str, timeframe: str, payload: dict[str, Any]) -> "Candle":
        """Build a Candle from a Capital.com ``/prices`` history entry.

        Capital.com returns bid/ask sub-objects for each OHLC point; we use the
        mid price (average of bid and ask) as the canonical price.
        """

        def mid(node: dict[str, Any]) -> float:
            bid = node.get("bid")
            ask = node.get("ask")
            if bid is not None and ask is not None:
                return (float(bid) + float(ask)) / 2.0
            return float(bid if bid is not None else ask)

        ts_raw = payload.get("snapshotTimeUTC") or payload["snapshotTime"]
        ts = _parse_ts(ts_raw)
        return cls(
            epic=epic,
            timeframe=timeframe,
            timestamp=ts,
            open=mid(payload["openPrice"]),
            high=mid(payload["highPrice"]),
            low=mid(payload["lowPrice"]),
            close=mid(payload["closePrice"]),
            volume=float(payload.get("lastTradedVolume", 0) or 0),
        )


@dataclass
class Signal:
    """A strategy's trading intent for one instrument at one point in time."""

    epic: str
    type: SignalType
    timestamp: datetime = field(default_factory=_utcnow)
    # Optional protective levels expressed as absolute prices.
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # Free-form annotations (indicator values, reason) for logging/audit.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.type in (SignalType.ENTER_LONG, SignalType.ENTER_SHORT)

    @property
    def side(self) -> Optional[Side]:
        if self.type is SignalType.ENTER_LONG:
            return Side.BUY
        if self.type is SignalType.ENTER_SHORT:
            return Side.SELL
        return None


@dataclass
class Order:
    """An order request / record."""

    epic: str
    side: Side
    size: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    deal_reference: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class Position:
    """An open position."""

    epic: str
    side: Side
    size: float
    entry_price: float
    deal_id: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=_utcnow)

    def unrealized_pnl(self, price: float, value_per_point: float = 1.0) -> float:
        """Mark-to-market PnL in account currency (before fees)."""
        return (price - self.entry_price) * self.side.sign * self.size * value_per_point

    @classmethod
    def from_capital(cls, payload: dict[str, Any]) -> "Position":
        pos = payload.get("position", payload)
        market = payload.get("market", {})
        return cls(
            epic=market.get("epic") or pos.get("epic"),
            side=Side(pos["direction"]),
            size=float(pos["size"]),
            entry_price=float(pos["level"]),
            deal_id=pos.get("dealId"),
            stop_loss=_opt_float(pos.get("stopLevel")),
            take_profit=_opt_float(pos.get("limitLevel")),
        )


@dataclass
class Trade:
    """A round-trip (closed) trade, used for reporting."""

    epic: str
    side: Side
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees: float = 0.0

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * self.side.sign

    @property
    def holding_period_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _parse_ts(raw: Any) -> datetime:
    """Parse a Capital.com timestamp (ISO string or epoch ms) to aware UTC."""
    if isinstance(raw, (int, float)):
        # Capital.com epochs are milliseconds.
        return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
    s = str(raw)
    # API may return "2022-01-31T00:00:00" (no tz) — treat as UTC.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

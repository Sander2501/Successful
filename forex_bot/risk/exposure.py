"""Currency-exposure decomposition for correlation-aware risk limits.

An FX position is really a bet on two currencies: long EURUSD is long EUR and
short USD. Several "different" pairs can therefore stack the same underlying
currency bet (EURUSD, GBPUSD and AUDUSD are all short USD). Decomposing each
position into per-currency notionals lets the risk manager cap that hidden
concentration instead of treating the pairs as independent.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from ..models import Position, Side

# A mapping from epic -> (base, quote) currency codes.
CurrencyMap = dict[str, tuple[str, str]]


def parse_currencies(
    epic: str, override: Optional[tuple[Optional[str], Optional[str]]] = None
) -> Optional[tuple[str, str]]:
    """Return (base, quote) for an FX epic, or None if it can't be determined.

    An explicit override wins; otherwise a clean 6-letter alphabetic epic is
    split into two 3-letter codes (``EURUSD`` -> ``("EUR", "USD")``).
    """
    if override and override[0] and override[1]:
        return override[0].upper(), override[1].upper()
    cleaned = epic.replace("/", "").replace("_", "").upper()
    if len(cleaned) == 6 and cleaned.isalpha():
        return cleaned[:3], cleaned[3:]
    return None


def notional(size: float, price: float, value_per_point: float = 1.0) -> float:
    """Position notional in quote-currency terms."""
    return abs(size) * price * value_per_point


def position_contributions(
    epic: str,
    side: Side,
    size: float,
    price: float,
    currency_map: CurrencyMap,
    *,
    value_per_point: float = 1.0,
) -> dict[str, float]:
    """Signed per-currency exposure for one position.

    Long the pair => +base, -quote; short => the reverse. Empty if the epic's
    currencies are unknown (the position is then ignored by currency checks).
    """
    pair = currency_map.get(epic)
    if pair is None:
        return {}
    base, quote = pair
    notion = notional(size, price, value_per_point)
    sign = 1.0 if side is Side.BUY else -1.0
    return {base: sign * notion, quote: -sign * notion}


def net_currency_exposures(
    positions: Iterable[Position],
    currency_map: CurrencyMap,
    *,
    value_per_point: float = 1.0,
    vpp_for: Optional["Callable[[str], float]"] = None,
) -> dict[str, float]:
    """Aggregate signed per-currency exposure across positions.

    ``vpp_for`` supplies a per-epic value-per-point (so a mixed-scale basket is
    priced correctly); when omitted, the single ``value_per_point`` is used.
    """
    totals: dict[str, float] = {}
    for pos in positions:
        vpp = vpp_for(pos.epic) if vpp_for is not None else value_per_point
        contrib = position_contributions(
            pos.epic, pos.side, pos.size, pos.entry_price, currency_map,
            value_per_point=vpp,
        )
        for ccy, amount in contrib.items():
            totals[ccy] = totals.get(ccy, 0.0) + amount
    return totals


def build_currency_map(instruments) -> CurrencyMap:
    """Build an epic -> (base, quote) map from InstrumentConfig entries."""
    out: CurrencyMap = {}
    for inst in instruments:
        pair = parse_currencies(
            inst.epic, (inst.base_currency, inst.quote_currency)
        )
        if pair is not None:
            out[inst.epic] = pair
    return out

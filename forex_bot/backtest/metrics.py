"""Performance metrics computed from an equity curve and trade list.

Pure stdlib (no numpy) so reporting runs anywhere. Ratios are annualized using
a supplied periods-per-year figure derived from the equity sampling frequency.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Sequence

from ..models import Trade


@dataclass
class PerformanceReport:
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    volatility_annual_pct: float
    num_trades: int
    win_rate_pct: float
    avg_trade_pnl: float
    profit_factor: float
    expectancy: float

    def as_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        rows = [
            ("Starting equity", f"{self.starting_equity:,.2f}"),
            ("Ending equity", f"{self.ending_equity:,.2f}"),
            ("Total return", f"{self.total_return_pct:.2f}%"),
            ("CAGR", f"{self.cagr_pct:.2f}%"),
            ("Max drawdown", f"{self.max_drawdown_pct:.2f}%"),
            ("Annual volatility", f"{self.volatility_annual_pct:.2f}%"),
            ("Sharpe", f"{self.sharpe:.2f}"),
            ("Sortino", f"{self.sortino:.2f}"),
            ("Trades", f"{self.num_trades}"),
            ("Win rate", f"{self.win_rate_pct:.2f}%"),
            ("Avg trade PnL", f"{self.avg_trade_pnl:,.2f}"),
            ("Profit factor", f"{self.profit_factor:.2f}"),
            ("Expectancy", f"{self.expectancy:,.2f}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k.ljust(width)} : {v}" for k, v in rows)


def compute_metrics(
    equity_curve: Sequence[tuple[datetime, float]],
    trades: Sequence[Trade],
    *,
    risk_free_rate: float = 0.0,
) -> PerformanceReport:
    if not equity_curve:
        raise ValueError("empty equity curve")

    equities = [e for _, e in equity_curve]
    start_eq = equities[0]
    end_eq = equities[-1]
    total_return = (end_eq / start_eq - 1.0) if start_eq else 0.0

    returns = _period_returns(equities)
    ppy = _periods_per_year(equity_curve)

    vol = _stdev(returns)
    mean_ret = sum(returns) / len(returns) if returns else 0.0
    rf_per_period = risk_free_rate / ppy if ppy else 0.0

    sharpe = ((mean_ret - rf_per_period) / vol * math.sqrt(ppy)) if vol > 0 else 0.0
    downside = _stdev([min(r - rf_per_period, 0.0) for r in returns], population=True)
    sortino = ((mean_ret - rf_per_period) / downside * math.sqrt(ppy)) if downside > 0 else 0.0
    vol_annual = vol * math.sqrt(ppy)

    years = _years_span(equity_curve)
    cagr = ((end_eq / start_eq) ** (1.0 / years) - 1.0) if start_eq > 0 and years > 0 else 0.0

    return PerformanceReport(
        starting_equity=start_eq,
        ending_equity=end_eq,
        total_return_pct=total_return * 100.0,
        cagr_pct=cagr * 100.0,
        max_drawdown_pct=_max_drawdown(equities) * 100.0,
        sharpe=sharpe,
        sortino=sortino,
        volatility_annual_pct=vol_annual * 100.0,
        num_trades=len(trades),
        win_rate_pct=_win_rate(trades) * 100.0,
        avg_trade_pnl=(sum(t.pnl for t in trades) / len(trades)) if trades else 0.0,
        profit_factor=_profit_factor(trades),
        expectancy=_expectancy(trades),
    )


# --------------------------------------------------------------------------- #
def _period_returns(equities: Sequence[float]) -> list[float]:
    out = []
    for prev, cur in zip(equities, equities[1:]):
        out.append((cur / prev - 1.0) if prev else 0.0)
    return out


def _stdev(values: Sequence[float], *, population: bool = False) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    denom = n if population else n - 1
    var = sum((v - mean) ** 2 for v in values) / denom
    return math.sqrt(var)


def _max_drawdown(equities: Sequence[float]) -> float:
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _periods_per_year(curve: Sequence[tuple[datetime, float]]) -> float:
    if len(curve) < 2:
        return 252.0
    spans = [
        (b[0] - a[0]).total_seconds()
        for a, b in zip(curve, curve[1:])
        if (b[0] - a[0]).total_seconds() > 0
    ]
    if not spans:
        return 252.0
    median = sorted(spans)[len(spans) // 2]
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / median


def _years_span(curve: Sequence[tuple[datetime, float]]) -> float:
    seconds = (curve[-1][0] - curve[0][0]).total_seconds()
    return seconds / (365.25 * 24 * 3600)


def _win_rate(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades)


def _profit_factor(trades: Sequence[Trade]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _expectancy(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(t.pnl for t in trades) / len(trades)

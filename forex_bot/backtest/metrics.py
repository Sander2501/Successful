"""Performance metrics computed from an equity curve and trade list.

Pure stdlib (no numpy/pandas) so reporting runs anywhere.

Return-based statistics (Sharpe, Sortino, volatility) are computed on the equity
curve **resampled to one point per calendar day**. This is deliberate: the raw
curve is sampled once per candle, so annualizing intra-day returns by their
native frequency wildly inflates volatility (a 15-minute bar implies ~35k
periods/year). Daily resampling with a 252-trading-day year gives figures that
are comparable to how strategies are normally quoted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Sequence

from ..models import Trade

#: Trading days per year used to annualize daily return statistics.
TRADING_DAYS_PER_YEAR = 252


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
    trading_days: int
    trades_per_day: float
    total_fees: float
    avg_fee_per_trade: float

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
            ("Sharpe (daily)", f"{self.sharpe:.2f}"),
            ("Sortino (daily)", f"{self.sortino:.2f}"),
            ("Trades", f"{self.num_trades}"),
            ("Trading days", f"{self.trading_days}"),
            ("Trades / day", f"{self.trades_per_day:.2f}"),
            ("Win rate", f"{self.win_rate_pct:.2f}%"),
            ("Avg trade PnL", f"{self.avg_trade_pnl:,.2f}"),
            ("Profit factor", f"{self.profit_factor:.2f}"),
            ("Expectancy", f"{self.expectancy:,.2f}"),
            ("Total fees", f"{self.total_fees:,.2f}"),
            ("Avg fee / trade", f"{self.avg_fee_per_trade:,.4f}"),
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

    # Daily-resampled returns drive the annualized risk statistics.
    daily = _resample_daily(equity_curve)
    daily_equities = [e for _, e in daily]
    daily_returns = _period_returns(daily_equities)
    trading_days = len(daily)

    vol = _stdev(daily_returns)
    mean_ret = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    ann = math.sqrt(TRADING_DAYS_PER_YEAR)

    sharpe = ((mean_ret - rf_daily) / vol * ann) if vol > 0 else 0.0
    downside = _stdev([min(r - rf_daily, 0.0) for r in daily_returns], population=True)
    sortino = ((mean_ret - rf_daily) / downside * ann) if downside > 0 else 0.0
    vol_annual = vol * ann

    years = _years_span(equity_curve)
    cagr = ((end_eq / start_eq) ** (1.0 / years) - 1.0) if start_eq > 0 and years > 0 else 0.0

    total_fees = sum(t.fees for t in trades)
    n = len(trades)
    return PerformanceReport(
        starting_equity=start_eq,
        ending_equity=end_eq,
        total_return_pct=total_return * 100.0,
        cagr_pct=cagr * 100.0,
        max_drawdown_pct=_max_drawdown(equities) * 100.0,
        sharpe=sharpe,
        sortino=sortino,
        volatility_annual_pct=vol_annual * 100.0,
        num_trades=n,
        win_rate_pct=_win_rate(trades) * 100.0,
        avg_trade_pnl=(sum(t.pnl for t in trades) / n) if n else 0.0,
        profit_factor=_profit_factor(trades),
        expectancy=_expectancy(trades),
        trading_days=trading_days,
        trades_per_day=(n / trading_days) if trading_days else 0.0,
        total_fees=total_fees,
        avg_fee_per_trade=(total_fees / n) if n else 0.0,
    )


# --------------------------------------------------------------------------- #
def _resample_daily(curve: Sequence[tuple[datetime, float]]) -> list[tuple[date, float]]:
    """Collapse the curve to the last equity value seen on each UTC day."""
    by_day: dict[date, float] = {}
    for ts, eq in curve:
        by_day[ts.date()] = eq  # later samples overwrite earlier ones
    return [(d, by_day[d]) for d in sorted(by_day)]


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

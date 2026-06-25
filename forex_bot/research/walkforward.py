"""Walk-forward optimization and out-of-sample evaluation.

This is the tool that makes "edge" a testable claim rather than an exercise in
curve fitting. The timeline is split into rolling folds; in each fold the
strategy parameters are optimized on an **in-sample** window, then evaluated on
the immediately following **out-of-sample** window the optimizer never saw. Only
the pooled out-of-sample performance is trusted.

Key anti-overfitting properties:
  * Parameters are always tested on unseen data.
  * Out-of-sample windows are warmed with prior bars (for indicator state) but
    only trades opened *inside* the OOS window count.
  * A ``min_trades`` floor stops the optimizer from "winning" in-sample on one
    or two lucky trades.

If the pooled OOS edge is positive and reasonably stable across folds, it is
worth taking seriously; if it evaporates out-of-sample, it was never real.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional, Sequence

from ..backtest.engine import Backtester
from ..backtest.metrics import PerformanceReport, compute_metrics
from ..config import TradingConfig
from ..models import Candle, Trade
from ..strategy import build_strategy

# A metric extractor scores a (report, trades) pair. Higher is better.
MetricFn = Callable[[PerformanceReport, Sequence[Trade]], float]

# Sensible default search grids per strategy, used by `optimize --all-strategies`
# (and as a fallback when the config has no strategy-specific grid).
DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    "ema_crossover": {
        "fast": [8, 12, 16],
        "slow": [26, 40, 60],
        "trend_filter": [100, 200],
        "adx_period": [14],
        "adx_threshold": [18.0, 25.0],
    },
    "donchian_breakout": {
        "entry": [20, 40, 55],
        "exit": [10, 20],
        "adx_period": [14],
        "adx_threshold": [18.0, 25.0],
    },
    "rsi_reversion": {
        "period": [7, 14],
        "oversold": [25.0, 30.0],
        "overbought": [70.0, 75.0],
    },
}


METRICS: dict[str, MetricFn] = {
    "sharpe": lambda r, t: r.sharpe,
    "sortino": lambda r, t: r.sortino,
    "total_return": lambda r, t: r.total_return_pct,
    "cagr": lambda r, t: r.cagr_pct,
    "profit_factor": lambda r, t: (r.profit_factor if r.profit_factor != float("inf") else 1e6),
    "expectancy": lambda r, t: r.expectancy,
}


@dataclass
class Fold:
    index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    best_params: dict[str, Any]
    is_metric: float
    oos_return_pct: float
    oos_trades: int
    oos_profit_factor: float
    oos_win_rate_pct: float


@dataclass
class WalkForwardResult:
    strategy: str
    metric: str
    folds: list[Fold] = field(default_factory=list)
    combined_oos_return_pct: float = 0.0
    avg_fold_return_pct: float = 0.0
    pct_positive_folds: float = 0.0
    total_oos_trades: int = 0
    combined_profit_factor: float = 0.0
    combined_win_rate_pct: float = 0.0

    def to_text(self) -> str:
        lines = [
            f"Walk-forward: {self.strategy}  (optimizing '{self.metric}')",
            f"  folds                 : {len(self.folds)}",
            f"  combined OOS return   : {self.combined_oos_return_pct:.2f}%",
            f"  avg fold OOS return   : {self.avg_fold_return_pct:.2f}%",
            f"  positive OOS folds    : {self.pct_positive_folds:.0f}%",
            f"  OOS trades (pooled)   : {self.total_oos_trades}",
            f"  OOS profit factor     : {self.combined_profit_factor:.2f}",
            f"  OOS win rate          : {self.combined_win_rate_pct:.2f}%",
            "",
            "  fold  in-sample -> out-of-sample        best params"
            "                          OOS ret%  trades  PF",
        ]
        for f in self.folds:
            lines.append(
                f"  {f.index:>4}  {f.is_start.date()}..{f.is_end.date()} -> "
                f"{f.oos_start.date()}..{f.oos_end.date()}  "
                f"{_fmt_params(f.best_params):<34}  "
                f"{f.oos_return_pct:>7.2f}  {f.oos_trades:>5}  {f.oos_profit_factor:>4.2f}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def param_combinations(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid into a list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def grid_search(
    candles_by_epic: dict[str, list[Candle]],
    config: TradingConfig,
    strategy_name: str,
    grid: dict[str, Sequence[Any]],
    *,
    metric: str = "sharpe",
    min_trades: int = 5,
) -> tuple[dict[str, Any], float]:
    """Return the (best_params, score) over ``grid`` on the given candles."""
    metric_fn = METRICS[metric]
    best_params: dict[str, Any] = {}
    best_score = float("-inf")
    for params in param_combinations(grid):
        report, trades, _ = _run_slice(candles_by_epic, config, strategy_name, params)
        if report is None or len(trades) < min_trades:
            continue
        score = metric_fn(report, trades)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def walk_forward(
    candles_by_epic: dict[str, list[Candle]],
    config: TradingConfig,
    strategy_name: str,
    grid: dict[str, Sequence[Any]],
    *,
    is_bars: int,
    oos_bars: int,
    step_bars: Optional[int] = None,
    metric: str = "sharpe",
    warmup_bars: int = 250,
    min_trades: int = 5,
) -> WalkForwardResult:
    """Run a rolling walk-forward optimization and return pooled OOS results."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}'; one of {sorted(METRICS)}")
    step_bars = step_bars or oos_bars

    timeline = sorted({c.timestamp for candles in candles_by_epic.values() for c in candles})
    n = len(timeline)
    if n < is_bars + oos_bars:
        raise ValueError(
            f"not enough data for walk-forward: need >= {is_bars + oos_bars} "
            f"timestamps, have {n}"
        )

    result = WalkForwardResult(strategy=strategy_name, metric=metric)
    pooled_oos: list[Trade] = []
    fold_returns: list[float] = []

    fold_idx = 0
    is_start_i = 0
    while is_start_i + is_bars + oos_bars <= n:
        is_end_i = is_start_i + is_bars
        oos_end_i = min(is_end_i + oos_bars, n)
        is_start_ts = timeline[is_start_i]
        oos_start_ts = timeline[is_end_i]
        oos_end_ts = timeline[oos_end_i - 1]
        warm_start_ts = timeline[max(0, is_end_i - warmup_bars)]

        is_slice = _slice(candles_by_epic, is_start_ts, oos_start_ts)  # [is_start, oos_start)
        best_params, is_score = grid_search(
            is_slice, config, strategy_name, grid, metric=metric, min_trades=min_trades
        )

        # Evaluate the chosen params on OOS data, warmed with prior bars.
        oos_slice = _slice(candles_by_epic, warm_start_ts, oos_end_ts, inclusive_end=True)
        report, trades, _ = _run_slice(oos_slice, config, strategy_name, best_params)

        oos_trades = [t for t in trades if t.entry_time >= oos_start_ts]
        oos_report, oos_ret = _oos_report(report, trades, oos_start_ts)

        pooled_oos.extend(oos_trades)
        fold_returns.append(oos_ret)
        result.folds.append(
            Fold(
                index=fold_idx,
                is_start=is_start_ts,
                is_end=timeline[is_end_i - 1],
                oos_start=oos_start_ts,
                oos_end=oos_end_ts,
                best_params=best_params,
                is_metric=is_score,
                oos_return_pct=oos_ret,
                oos_trades=len(oos_trades),
                oos_profit_factor=oos_report.profit_factor if oos_report else 0.0,
                oos_win_rate_pct=oos_report.win_rate_pct if oos_report else 0.0,
            )
        )
        fold_idx += 1
        is_start_i += step_bars

    _summarize(result, pooled_oos, fold_returns)
    return result


# --------------------------------------------------------------------------- #
def _slice(
    candles_by_epic: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
    *,
    inclusive_end: bool = False,
) -> dict[str, list[Candle]]:
    out: dict[str, list[Candle]] = {}
    for epic, candles in candles_by_epic.items():
        if inclusive_end:
            sel = [c for c in candles if start <= c.timestamp <= end]
        else:
            sel = [c for c in candles if start <= c.timestamp < end]
        if sel:
            out[epic] = sel
    return out


def _run_slice(
    candles_by_epic: dict[str, list[Candle]],
    config: TradingConfig,
    strategy_name: str,
    params: dict[str, Any],
) -> tuple[Optional[PerformanceReport], list[Trade], Any]:
    if not candles_by_epic:
        return None, [], None
    try:
        strategy = build_strategy(strategy_name, params)
        bt = Backtester(strategy, config)
        res = bt.run({k: list(v) for k, v in candles_by_epic.items()})
    except Exception:
        return None, [], None
    if not res.equity_curve:
        return None, res.trades, res
    report = compute_metrics(res.equity_curve, res.trades)
    return report, res.trades, res


def _oos_report(
    full_report: Optional[PerformanceReport],
    trades: Sequence[Trade],
    oos_start: datetime,
) -> tuple[Optional[PerformanceReport], float]:
    """Build a report restricted to OOS trades and compute the OOS return."""
    oos_trades = [t for t in trades if t.entry_time >= oos_start]
    if not oos_trades:
        return None, 0.0
    # Return is the summed trade PnL over the starting equity (cash-based, robust
    # to the warmup portion of the equity curve).
    start_eq = full_report.starting_equity if full_report else 10_000.0
    pnl = sum(t.pnl for t in oos_trades)
    oos_ret = (pnl / start_eq) * 100.0 if start_eq else 0.0
    # Reuse metric helpers via a synthetic single-point equity curve.
    from ..backtest.metrics import _profit_factor, _win_rate  # local import

    report = PerformanceReport(
        starting_equity=start_eq,
        ending_equity=start_eq + pnl,
        total_return_pct=oos_ret,
        cagr_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe=0.0,
        sortino=0.0,
        volatility_annual_pct=0.0,
        num_trades=len(oos_trades),
        win_rate_pct=_win_rate(oos_trades) * 100.0,
        avg_trade_pnl=pnl / len(oos_trades),
        profit_factor=_profit_factor(oos_trades),
        expectancy=pnl / len(oos_trades),
        trading_days=0,
        trades_per_day=0.0,
        total_fees=sum(t.fees for t in oos_trades),
        avg_fee_per_trade=0.0,
    )
    return report, oos_ret


def _summarize(
    result: WalkForwardResult, pooled: list[Trade], fold_returns: list[float]
) -> None:
    from ..backtest.metrics import _profit_factor, _win_rate

    result.total_oos_trades = len(pooled)
    result.combined_profit_factor = _profit_factor(pooled)
    result.combined_win_rate_pct = _win_rate(pooled) * 100.0
    if fold_returns:
        # Chain per-fold returns to a compounded combined OOS return.
        compounded = 1.0
        for r in fold_returns:
            compounded *= 1.0 + r / 100.0
        result.combined_oos_return_pct = (compounded - 1.0) * 100.0
        result.avg_fold_return_pct = sum(fold_returns) / len(fold_returns)
        positive = sum(1 for r in fold_returns if r > 0)
        result.pct_positive_folds = positive / len(fold_returns) * 100.0


def _fmt_params(params: dict[str, Any]) -> str:
    if not params:
        return "(defaults)"
    return ",".join(f"{k}={v}" for k, v in params.items())

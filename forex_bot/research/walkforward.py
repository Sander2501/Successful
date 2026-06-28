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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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
        "atr_regime_lookback": [None, 100],
        "atr_regime_quantile": [0.6],
    },
    "rsi_reversion": {
        "period": [7, 14],
        "oversold": [25.0, 30.0],
        "overbought": [70.0, 75.0],
    },
    "spread_reversion": {
        "lookback": [60, 100, 150],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.25, 0.5],
    },
    "sweep_reversal": {
        # min_rr starts at/above breakeven for a low-win-rate reversal; the
        # sweep-quality gate and sweep-side FVG are the structural fixes from the
        # review. Session filtering is left to config (gridding hours is noisy).
        "swing_k": [2, 3],
        "lookback": [40, 60],
        "min_rr": [2.5, 3.0],
        "min_pen_atr": [0.0, 0.25],
        "min_rej_frac": [0.5],
        "fvg_prefer": ["sweep"],
    },
    "xsec_momentum": {
        "lookback": [50, 100],
        "top_k": [1, 2],
        "rebalance_bars": [10, 20],
    },
    "carry_trend_basket": {
        "lookback": [125, 250],
        "top_k": [1, 2],
        "bottom_k": [1],
        "rebalance_bars": [20, 40],
        "min_abs_trend": [0.01, 0.02],
        "carry_weight": [0.0],
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
    selected_epics: list[str] = field(default_factory=list)


@dataclass
class PeriodBreakdown:
    period: str
    pnl: float = 0.0
    return_pct: float = 0.0
    trades: int = 0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    avg_r_multiple: float = 0.0
    avg_holding_hours: float = 0.0


@dataclass
class InstrumentBreakdown:
    epic: str
    pnl: float = 0.0
    return_pct: float = 0.0
    trades: int = 0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    avg_r_multiple: float = 0.0
    avg_holding_hours: float = 0.0


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
    by_instrument: list[InstrumentBreakdown] = field(default_factory=list)
    by_period: list[PeriodBreakdown] = field(default_factory=list)
    starting_equity: float = 10_000.0

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
            selected = ""
            if f.selected_epics:
                selected = f"  selected={','.join(f.selected_epics)}"
            lines.append(
                f"  {f.index:>4}  {f.is_start.date()}..{f.is_end.date()} -> "
                f"{f.oos_start.date()}..{f.oos_end.date()}  "
                f"{_fmt_params(f.best_params):<34}  "
                f"{f.oos_return_pct:>7.2f}  {f.oos_trades:>5}  {f.oos_profit_factor:>4.2f}"
                f"{selected}"
            )
        if self.by_instrument:
            lines.extend([
                "",
                "  instrument breakdown (pooled OOS)",
                f"  {'epic':<10} {'ret%':>8} {'trades':>7} {'PF':>6} "
                f"{'win%':>7} {'avg R':>7} {'hold h':>7}",
            ])
            for b in self.by_instrument:
                lines.append(
                    f"  {b.epic:<10} {b.return_pct:>8.2f} {b.trades:>7} "
                    f"{b.profit_factor:>6.2f} {b.win_rate_pct:>7.1f} "
                    f"{b.avg_r_multiple:>7.2f} {b.avg_holding_hours:>7.1f}"
                )
        if self.by_period:
            lines.extend([
                "",
                "  monthly breakdown (pooled OOS)",
                f"  {'month':<10} {'ret%':>8} {'trades':>7} {'PF':>6} "
                f"{'win%':>7} {'avg R':>7} {'hold h':>7}",
            ])
            for b in self.by_period:
                lines.append(
                    f"  {b.period:<10} {b.return_pct:>8.2f} {b.trades:>7} "
                    f"{b.profit_factor:>6.2f} {b.win_rate_pct:>7.1f} "
                    f"{b.avg_r_multiple:>7.2f} {b.avg_holding_hours:>7.1f}"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
@dataclass
class HoldoutResult:
    """Outcome of a single, final test on data never used for optimization."""

    strategy: str
    best_params: dict[str, Any]
    train_metric: float
    holdout_return_pct: float
    holdout_trades: int
    holdout_profit_factor: float
    holdout_win_rate_pct: float
    train_start: datetime
    train_end: datetime
    holdout_start: datetime
    holdout_end: datetime

    def to_text(self) -> str:
        return "\n".join([
            f"One-shot holdout - {self.strategy}",
            f"  trained on : {self.train_start.date()}..{self.train_end.date()} "
            f"(best params {_fmt_params(self.best_params)})",
            f"  held out   : {self.holdout_start.date()}..{self.holdout_end.date()} "
            f"(never seen during optimization)",
            "",
            f"  HOLDOUT return     : {self.holdout_return_pct:.2f}%",
            f"  HOLDOUT trades     : {self.holdout_trades}",
            f"  HOLDOUT profit factor: {self.holdout_profit_factor:.2f}",
            f"  HOLDOUT win rate   : {self.holdout_win_rate_pct:.2f}%",
        ])


def holdout_test(
    candles_by_epic: dict[str, list[Candle]],
    config: TradingConfig,
    strategy_name: str,
    grid: dict[str, Sequence[Any]],
    *,
    holdout_frac: float = 0.2,
    metric: str = "sharpe",
    warmup_bars: int = 250,
    min_trades: int = 5,
    cost_multiplier: float = 1.0,
) -> HoldoutResult:
    """Optimize on the first ``1 - holdout_frac`` of the data and test ONCE on the
    final ``holdout_frac`` that optimization never touched.

    This is the antidote to data-snooping: parameters are chosen without ever
    seeing the holdout, so the holdout result is a genuinely out-of-sample read.
    Run it exactly once - re-running and re-tuning defeats the purpose.
    """
    if not (0.05 <= holdout_frac <= 0.5):
        raise ValueError("holdout_frac should be between 0.05 and 0.5")
    config = _scaled_costs(config, cost_multiplier)
    timeline = sorted({c.timestamp for candles in candles_by_epic.values() for c in candles})
    n = len(timeline)
    split = int(n * (1.0 - holdout_frac))
    if n < 50 or split < 10 or split >= n - 1:
        raise ValueError(f"not enough data for a holdout split (have {n} bars)")
    split_ts = timeline[split]
    warm_start_ts = timeline[max(0, split - warmup_bars)]

    train = _slice(candles_by_epic, timeline[0], split_ts)              # [start, split)
    holdout = _slice(candles_by_epic, warm_start_ts, timeline[-1], inclusive_end=True)

    best_params, train_score = grid_search(
        train, config, strategy_name, grid, metric=metric, min_trades=min_trades
    )
    report, trades, _ = _run_slice(holdout, config, strategy_name, best_params)
    ho_report, ho_ret = _oos_report(report, trades, split_ts)
    return HoldoutResult(
        strategy=strategy_name,
        best_params=best_params,
        train_metric=train_score,
        holdout_return_pct=ho_ret,
        holdout_trades=ho_report.num_trades if ho_report else 0,
        holdout_profit_factor=ho_report.profit_factor if ho_report else 0.0,
        holdout_win_rate_pct=ho_report.win_rate_pct if ho_report else 0.0,
        train_start=timeline[0],
        train_end=timeline[split - 1],
        holdout_start=split_ts,
        holdout_end=timeline[-1],
    )


def _scaled_costs(config: TradingConfig, multiplier: float) -> TradingConfig:
    """Return a config copy with trading costs scaled by ``multiplier``.

    Scales BOTH the global ``costs.spread_points`` fallback AND each instrument's
    own ``spread_points``. The latter is essential: the executor charges the
    per-instrument spread when one is configured (see ``SimulatedExecution`` and
    ``InstrumentSpecs``), so scaling only the global fallback would leave the
    cost-stress test charging 1x spread on every configured instrument - making
    a spread-thin edge look like it survives 2-3x costs when it does not.
    """
    if multiplier == 1.0:
        return config
    import dataclasses

    from ..config import CostConfig
    c = config.costs
    scaled_instruments = [
        dataclasses.replace(
            inst,
            spread_points=(inst.spread_points * multiplier
                           if inst.spread_points is not None else None),
        )
        for inst in config.instruments
    ]
    return dataclasses.replace(
        config,
        costs=CostConfig(
            spread_points=c.spread_points * multiplier,
            commission_per_trade=c.commission_per_trade * multiplier,
            slippage_points=c.slippage_points * multiplier,
        ),
        instruments=scaled_instruments,
    )


def param_combinations(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid into a list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


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
    step_bars: int | None = None,
    metric: str = "sharpe",
    warmup_bars: int = 250,
    min_trades: int = 5,
    cost_multiplier: float = 1.0,
    select_top_n: int | None = None,
    select_metric: str = "return",
) -> WalkForwardResult:
    """Run a rolling walk-forward optimization and return pooled OOS results.

    ``cost_multiplier`` scales the configured spread/slippage/commission so the
    same edge can be re-evaluated under heavier (more realistic) costs.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}'; one of {sorted(METRICS)}")
    step_bars = step_bars or oos_bars
    config = _scaled_costs(config, cost_multiplier)

    timeline = sorted({c.timestamp for candles in candles_by_epic.values() for c in candles})
    n = len(timeline)
    if n < is_bars + oos_bars:
        raise ValueError(
            f"not enough data for walk-forward: need >= {is_bars + oos_bars} "
            f"timestamps, have {n}"
        )

    result = WalkForwardResult(strategy=strategy_name, metric=metric,
                               starting_equity=config.starting_equity)
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
        selected_epics: list[str] = []
        if select_top_n is not None:
            selected_epics = _select_top_epics(
                is_slice, config, strategy_name, grid,
                top_n=select_top_n, metric=select_metric,
                score_metric=metric, min_trades=min_trades,
            )
            if selected_epics:
                is_slice = {e: is_slice[e] for e in selected_epics if e in is_slice}

        best_params, is_score = grid_search(
            is_slice, config, strategy_name, grid, metric=metric, min_trades=min_trades
        )

        # Evaluate the chosen params on OOS data, warmed with prior bars.
        oos_slice = _slice(candles_by_epic, warm_start_ts, oos_end_ts, inclusive_end=True)
        if selected_epics:
            oos_slice = {e: oos_slice[e] for e in selected_epics if e in oos_slice}
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
                selected_epics=selected_epics,
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
) -> tuple[PerformanceReport | None, list[Trade], Any]:
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
    full_report: PerformanceReport | None,
    trades: Sequence[Trade],
    oos_start: datetime,
) -> tuple[PerformanceReport | None, float]:
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
    from ..backtest.metrics import (  # local import
        _avg_holding_hours,
        _avg_r_multiple,
        _profit_factor,
        _win_rate,
    )

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
        avg_r_multiple=_avg_r_multiple(oos_trades),
        avg_holding_hours=_avg_holding_hours(oos_trades),
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
    result.by_instrument = _instrument_breakdown(pooled, result.starting_equity)
    result.by_period = _period_breakdown(pooled, result.starting_equity)
    if fold_returns:
        # Chain per-fold returns to a compounded combined OOS return.
        compounded = 1.0
        for r in fold_returns:
            compounded *= 1.0 + r / 100.0
        result.combined_oos_return_pct = (compounded - 1.0) * 100.0
        result.avg_fold_return_pct = sum(fold_returns) / len(fold_returns)
        positive = sum(1 for r in fold_returns if r > 0)
        result.pct_positive_folds = positive / len(fold_returns) * 100.0



def _select_top_epics(
    is_slice: dict[str, list[Candle]],
    config: TradingConfig,
    strategy_name: str,
    grid: dict[str, Sequence[Any]],
    *,
    top_n: int,
    metric: str,
    score_metric: str,
    min_trades: int,
) -> list[str]:
    """Select instruments using only the current in-sample window."""
    if top_n <= 0 or len(is_slice) <= top_n:
        return sorted(is_slice)
    best_params, _score = grid_search(
        is_slice, config, strategy_name, grid, metric=score_metric, min_trades=min_trades
    )
    _report, trades, _res = _run_slice(is_slice, config, strategy_name, best_params)
    breakdown = _instrument_breakdown(trades, config.starting_equity)
    if not breakdown:
        return sorted(is_slice)[:top_n]

    def score(item: InstrumentBreakdown) -> float:
        if metric == "profit_factor":
            return item.profit_factor
        if metric == "trades":
            return float(item.trades)
        if metric == "expectancy":
            return item.pnl / item.trades if item.trades else float("-inf")
        return item.return_pct

    ranked = sorted(breakdown, key=score, reverse=True)
    selected = [b.epic for b in ranked[:top_n]]
    if len(selected) < top_n:
        for epic in sorted(is_slice):
            if epic not in selected:
                selected.append(epic)
            if len(selected) >= top_n:
                break
    return selected


def _instrument_breakdown(
    pooled: Sequence[Trade], starting_equity: float
) -> list[InstrumentBreakdown]:
    """Summarize pooled OOS trades per instrument."""
    from ..backtest.metrics import _avg_holding_hours, _avg_r_multiple, _profit_factor, _win_rate

    by_epic: dict[str, list[Trade]] = {}
    for trade in pooled:
        by_epic.setdefault(trade.epic, []).append(trade)

    out: list[InstrumentBreakdown] = []
    for epic, trades in by_epic.items():
        pnl = sum(t.pnl for t in trades)
        out.append(
            InstrumentBreakdown(
                epic=epic,
                pnl=pnl,
                return_pct=(pnl / starting_equity) * 100.0 if starting_equity else 0.0,
                trades=len(trades),
                profit_factor=_profit_factor(trades),
                win_rate_pct=_win_rate(trades) * 100.0,
                avg_r_multiple=_avg_r_multiple(trades),
                avg_holding_hours=_avg_holding_hours(trades),
            )
        )
    return sorted(out, key=lambda b: b.return_pct, reverse=True)


def _period_breakdown(
    pooled: Sequence[Trade], starting_equity: float
) -> list[PeriodBreakdown]:
    """Summarize pooled OOS trades by entry month."""
    from ..backtest.metrics import _avg_holding_hours, _avg_r_multiple, _profit_factor, _win_rate

    by_month: dict[str, list[Trade]] = {}
    for trade in pooled:
        key = trade.entry_time.strftime("%Y-%m")
        by_month.setdefault(key, []).append(trade)

    out: list[PeriodBreakdown] = []
    for month in sorted(by_month):
        trades = by_month[month]
        pnl = sum(t.pnl for t in trades)
        out.append(
            PeriodBreakdown(
                period=month,
                pnl=pnl,
                return_pct=(pnl / starting_equity) * 100.0 if starting_equity else 0.0,
                trades=len(trades),
                profit_factor=_profit_factor(trades),
                win_rate_pct=_win_rate(trades) * 100.0,
                avg_r_multiple=_avg_r_multiple(trades),
                avg_holding_hours=_avg_holding_hours(trades),
            )
        )
    return out



def _fmt_params(params: dict[str, Any]) -> str:
    if not params:
        return "(defaults)"
    return ",".join(f"{k}={v}" for k, v in params.items())

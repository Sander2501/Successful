"""Canonical strategy comparison / rejection report.

Runs every requested strategy through the *same* walk-forward configuration and
reduces each run to a single comparable row: OOS return, profit factor, positive
folds, trades, avg R, worst month, worst fold, per-instrument contribution, and a
deterministic pass/fail verdict (see ``verdicts.py``).

This replaces ad-hoc per-strategy text dumps with one standardized table that can
be diffed across research runs. The reducer is pure; only ``run_strategy_report``
executes walk-forward, and only the ``to_csv``/``to_markdown`` helpers format.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..config import TradingConfig
from ..models import Candle
from . import verdicts
from .verdicts import VerdictThresholds
from .walkforward import WalkForwardResult, walk_forward

# Column order shared by the CSV writer and the Markdown table.
COLUMNS = [
    "strategy",
    "oos_return_pct",
    "profit_factor",
    "pct_positive_folds",
    "total_trades",
    "avg_r_multiple",
    "worst_month_return_pct",
    "worst_fold_return_pct",
    "instrument_contribution",
    "verdict",
    "failed_rules",
]


@dataclass
class StrategyRow:
    strategy: str
    oos_return_pct: float
    profit_factor: float
    pct_positive_folds: float
    total_trades: int
    avg_r_multiple: float
    worst_month_return_pct: float
    worst_fold_return_pct: float
    instrument_contribution: str
    passed: bool
    failed_rules: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class StrategyReport:
    rows: list[StrategyRow] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _fmt_pf(pf: float) -> str:
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def _instrument_contribution(wf: WalkForwardResult) -> str:
    """Compact per-instrument summary, e.g. ``EURUSD:+3.1%(12t); GBPUSD:-0.4%(8t)``.

    ``by_instrument`` is already sorted by return descending upstream.
    """
    return "; ".join(
        f"{b.epic}:{b.return_pct:+.1f}%({b.trades}t)" for b in wf.by_instrument
    )


def build_row(wf: WalkForwardResult, thresholds: VerdictThresholds) -> StrategyRow:
    """Reduce one walk-forward result to a single comparable, judged row."""
    worst_fold = min((f.oos_return_pct for f in wf.folds), default=0.0)
    worst_month = min((b.return_pct for b in wf.by_period), default=0.0)
    instrument_returns = [b.return_pct for b in wf.by_instrument]

    verdict = verdicts.evaluate(
        oos_return_pct=wf.combined_oos_return_pct,
        profit_factor=wf.combined_profit_factor,
        pct_positive_folds=wf.pct_positive_folds,
        total_trades=wf.total_oos_trades,
        avg_r_multiple=wf.combined_avg_r_multiple,
        r_multiple_trades=wf.r_multiple_trades,
        worst_fold_return_pct=worst_fold,
        worst_month_return_pct=worst_month,
        instrument_returns=instrument_returns,
        thresholds=thresholds,
    )
    return StrategyRow(
        strategy=wf.strategy,
        oos_return_pct=wf.combined_oos_return_pct,
        profit_factor=wf.combined_profit_factor,
        pct_positive_folds=wf.pct_positive_folds,
        total_trades=wf.total_oos_trades,
        avg_r_multiple=wf.combined_avg_r_multiple,
        worst_month_return_pct=worst_month,
        worst_fold_return_pct=worst_fold,
        instrument_contribution=_instrument_contribution(wf),
        passed=verdict.passed,
        failed_rules=verdict.failed_rules,
    )


def run_strategy_report(
    candles_by_epic: dict[str, list[Candle]],
    config: TradingConfig,
    strategies: Sequence[str],
    grid_for: Callable[[str], dict],
    wf_kwargs: dict[str, Any],
    thresholds: VerdictThresholds,
    *,
    on_skip: Callable[[str, str], None] | None = None,
) -> StrategyReport:
    """Run walk-forward for each strategy under identical settings and reduce.

    Strategies with no parameter grid or insufficient data are skipped (reported
    via ``on_skip(name, reason)`` if provided) rather than aborting the report.
    Rows are sorted by OOS return, descending.
    """
    rows: list[StrategyRow] = []
    for name in strategies:
        grid = grid_for(name)
        if not grid:
            if on_skip:
                on_skip(name, "no parameter grid")
            continue
        try:
            wf = walk_forward(candles_by_epic, config, name, grid, **wf_kwargs)
        except ValueError as exc:
            if on_skip:
                on_skip(name, str(exc))
            continue
        rows.append(build_row(wf, thresholds))

    rows.sort(key=lambda r: r.oos_return_pct, reverse=True)

    timeline = sorted({c.timestamp for cs in candles_by_epic.values() for c in cs})
    meta = {
        "generated": date.today().isoformat(),
        "instruments": sorted(candles_by_epic),
        "data_start": timeline[0].date().isoformat() if timeline else "",
        "data_end": timeline[-1].date().isoformat() if timeline else "",
        "is_bars": wf_kwargs.get("is_bars"),
        "oos_bars": wf_kwargs.get("oos_bars"),
        "step_bars": wf_kwargs.get("step_bars"),
        "metric": wf_kwargs.get("metric"),
        "min_trades": wf_kwargs.get("min_trades"),
        "n_strategies": len(rows),
    }
    return StrategyReport(rows=rows, meta=meta)


def _row_cells(r: StrategyRow) -> list[str]:
    return [
        r.strategy,
        f"{r.oos_return_pct:.2f}",
        _fmt_pf(r.profit_factor),
        f"{r.pct_positive_folds:.0f}",
        str(r.total_trades),
        f"{r.avg_r_multiple:.2f}",
        f"{r.worst_month_return_pct:.2f}",
        f"{r.worst_fold_return_pct:.2f}",
        r.instrument_contribution,
        r.verdict,
        " | ".join(r.failed_rules),
    ]


def to_csv(report: StrategyReport) -> str:
    """Render the report as a clean tabular CSV (one row per strategy)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    for r in report.rows:
        writer.writerow(_row_cells(r))
    return buf.getvalue()


def to_markdown(report: StrategyReport) -> str:
    """Render the report as a Markdown doc: provenance header, table, verdict."""
    m = report.meta
    lines = [
        "# Strategy rejection report",
        "",
        f"- generated: {m.get('generated', '')}",
        f"- instruments: {', '.join(m.get('instruments', []))}",
        f"- data range: {m.get('data_start', '')} .. {m.get('data_end', '')}",
        f"- walk-forward: is={m.get('is_bars')} oos={m.get('oos_bars')} "
        f"step={m.get('step_bars')} metric={m.get('metric')} "
        f"min_trades={m.get('min_trades')}",
        "",
    ]
    header = (
        "| strategy | OOS ret% | PF | +folds% | trades | avg R | "
        "worst mo% | worst fold% | instrument contribution | verdict | reasons |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines.append(header)
    lines.append(sep)
    for r in report.rows:
        cells = [c.replace("|", "\\|") for c in _row_cells(r)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    passers = [r.strategy for r in report.rows if r.passed]
    if passers:
        lines.append(f"**PASS:** {', '.join(passers)} — worth a closer look.")
    else:
        lines.append(
            "**No strategy passed.** Do not trade live on this data; the "
            "out-of-sample edge was not robust under the rejection rules."
        )
    return "\n".join(lines)

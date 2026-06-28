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
import json
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
    "oos_return",
    "pf",
    "positive_folds",
    "trades",
    "avg_r",
    "worst_month",
    "worst_fold",
    "instrument_contribution",
    "verdict",
    "reasons",
]


@dataclass
class StrategyRow:
    strategy: str
    oos_return_pct: float
    profit_factor: float
    positive_folds: int
    total_folds: int
    total_trades: int
    avg_r_multiple: float
    r_multiple_trades: int  # 0 => avg R undefined (no stops) => shown as N/A
    worst_month_return_pct: float
    worst_month_label: str
    worst_fold_return_pct: float
    worst_fold_index: int
    instrument_contribution: str  # JSON: {epic: return_pct}
    passed: bool
    failed_rules: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def pct_positive_folds(self) -> float:
        return 100.0 * self.positive_folds / self.total_folds if self.total_folds else 0.0

    @property
    def folds_label(self) -> str:
        return f"{self.positive_folds}/{self.total_folds}"

    @property
    def avg_r_display(self) -> str:
        # Undefined when no trade carried a stop — say so rather than print 0.00.
        return "N/A" if self.r_multiple_trades == 0 else f"{self.avg_r_multiple:.2f}"

    @property
    def worst_month_display(self) -> str:
        label = self.worst_month_label or "n/a"
        return f"{label} ({self.worst_month_return_pct:.2f}%)"

    @property
    def worst_fold_display(self) -> str:
        return f"{self.worst_fold_index} ({self.worst_fold_return_pct:.2f}%)"


@dataclass
class StrategyReport:
    rows: list[StrategyRow] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _fmt_pf(pf: float) -> str:
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def _instrument_contribution(wf: WalkForwardResult) -> str:
    """Per-instrument OOS contribution as JSON ``{epic: return_pct}``.

    ``by_instrument`` is already sorted by return descending upstream, and dict
    insertion order is preserved, so the JSON reads best-to-worst.
    """
    return json.dumps({b.epic: round(b.return_pct, 4) for b in wf.by_instrument})


def build_row(wf: WalkForwardResult, thresholds: VerdictThresholds) -> StrategyRow:
    """Reduce one walk-forward result to a single comparable, judged row."""
    worst_fold_f = min(wf.folds, key=lambda f: f.oos_return_pct, default=None)
    worst_month_p = min(wf.by_period, key=lambda b: b.return_pct, default=None)
    instrument_returns = [b.return_pct for b in wf.by_instrument]

    verdict = verdicts.evaluate(
        oos_return_pct=wf.combined_oos_return_pct,
        profit_factor=wf.combined_profit_factor,
        pct_positive_folds=wf.pct_positive_folds,
        total_trades=wf.total_oos_trades,
        avg_r_multiple=wf.combined_avg_r_multiple,
        r_multiple_trades=wf.r_multiple_trades,
        worst_fold_return_pct=worst_fold_f.oos_return_pct if worst_fold_f else 0.0,
        worst_month_return_pct=worst_month_p.return_pct if worst_month_p else 0.0,
        instrument_returns=instrument_returns,
        thresholds=thresholds,
    )
    return StrategyRow(
        strategy=wf.strategy,
        oos_return_pct=wf.combined_oos_return_pct,
        profit_factor=wf.combined_profit_factor,
        positive_folds=sum(1 for f in wf.folds if f.oos_return_pct > 0),
        total_folds=len(wf.folds),
        total_trades=wf.total_oos_trades,
        avg_r_multiple=wf.combined_avg_r_multiple,
        r_multiple_trades=wf.r_multiple_trades,
        worst_month_return_pct=worst_month_p.return_pct if worst_month_p else 0.0,
        worst_month_label=worst_month_p.period if worst_month_p else "",
        worst_fold_return_pct=worst_fold_f.oos_return_pct if worst_fold_f else 0.0,
        worst_fold_index=worst_fold_f.index if worst_fold_f else 0,
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
        f"{r.oos_return_pct:.2f}%",
        _fmt_pf(r.profit_factor),
        r.folds_label,
        str(r.total_trades),
        r.avg_r_display,
        r.worst_month_display,
        r.worst_fold_display,
        r.instrument_contribution,
        r.verdict,
        "; ".join(r.failed_rules),
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
        "| strategy | OOS return | PF | + folds | trades | avg R | "
        "worst month | worst fold | instrument contribution | verdict | reasons |"
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

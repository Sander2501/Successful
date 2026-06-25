"""Export backtest artifacts: trade log (CSV), equity curve (CSV), HTML report.

Pure stdlib so reports generate anywhere. matplotlib is used only if available
to also drop a PNG equity chart.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

from ..models import Trade
from .engine import BacktestResult
from .metrics import PerformanceReport


def export_trades_csv(trades: list[Trade], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epic", "side", "size", "entry_price", "exit_price",
                    "entry_time", "exit_time", "pnl", "fees", "return_pct"])
        for t in trades:
            w.writerow([t.epic, t.side.value, t.size, t.entry_price, t.exit_price,
                        t.entry_time.isoformat(), t.exit_time.isoformat(),
                        round(t.pnl, 4), round(t.fees, 4), round(t.return_pct, 6)])
    return path


def export_equity_csv(equity_curve: list[tuple], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "equity"])
        for ts, eq in equity_curve:
            w.writerow([ts.isoformat(), round(eq, 4)])
    return path


def export_html_report(
    result: BacktestResult, report: PerformanceReport, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in report.as_dict().items()
    )
    trade_rows = "".join(
        f"<tr><td>{html.escape(t.epic)}</td><td>{t.side.value}</td>"
        f"<td>{t.size:.4f}</td><td>{t.entry_price:.5f}</td><td>{t.exit_price:.5f}</td>"
        f"<td>{t.entry_time.isoformat()}</td><td>{t.exit_time.isoformat()}</td>"
        f"<td>{t.pnl:.2f}</td></tr>"
        for t in result.trades
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Backtest report</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1b1b1b; }}
 h1 {{ margin-bottom: .25rem; }}
 table {{ border-collapse: collapse; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: right; }}
 th {{ background: #f5f5f5; text-align: left; }}
</style></head><body>
<h1>Backtest report</h1>
<p>Strategy: <strong>{html.escape(result.meta.get('strategy', '?'))}</strong> &middot;
   Signals: {result.signals_emitted} &middot; Fills: {result.orders_filled}</p>
<h2>Performance</h2>
<table>{metrics_rows}</table>
<h2>Trades ({len(result.trades)})</h2>
<table>
<tr><th>Epic</th><th>Side</th><th>Size</th><th>Entry</th><th>Exit</th>
    <th>Entry time</th><th>Exit time</th><th>PnL</th></tr>
{trade_rows}
</table>
</body></html>"""
    path.write_text(doc)
    return path

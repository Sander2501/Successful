import csv
import io
import unittest
from datetime import datetime, timezone

from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.research.strategy_report import (
    COLUMNS,
    StrategyReport,
    StrategyRow,
    build_row,
    run_strategy_report,
    to_csv,
    to_markdown,
)
from forex_bot.research.verdicts import VerdictThresholds
from forex_bot.research.walkforward import (
    Fold,
    InstrumentBreakdown,
    PeriodBreakdown,
    WalkForwardResult,
)
from tests.helpers import make_candles

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _fold(ret: float, idx: int = 0) -> Fold:
    return Fold(
        index=idx, is_start=_T0, is_end=_T0, oos_start=_T0, oos_end=_T0,
        best_params={}, is_metric=0.0, oos_return_pct=ret, oos_trades=5,
        oos_profit_factor=1.2, oos_win_rate_pct=50.0,
    )


def _instr(epic: str, ret: float, trades: int = 10) -> InstrumentBreakdown:
    return InstrumentBreakdown(epic=epic, return_pct=ret, trades=trades,
                               profit_factor=1.2, win_rate_pct=50.0,
                               avg_r_multiple=0.2, avg_holding_hours=10.0)


def _period(period: str, ret: float) -> PeriodBreakdown:
    return PeriodBreakdown(period=period, return_pct=ret, trades=5,
                           profit_factor=1.1, win_rate_pct=50.0,
                           avg_r_multiple=0.1, avg_holding_hours=8.0)


class TestBuildRow(unittest.TestCase):
    def _wf(self, **kw) -> WalkForwardResult:
        base = dict(
            strategy="s1", metric="sharpe",
            combined_oos_return_pct=4.0, pct_positive_folds=60.0,
            total_oos_trades=50, combined_profit_factor=1.4,
            combined_avg_r_multiple=0.25, r_multiple_trades=50,
            folds=[_fold(-2.0, 0), _fold(3.0, 1), _fold(-5.0, 2)],
            by_instrument=[_instr("EURUSD", 3.0, 30), _instr("GBPUSD", 1.0, 20)],
            by_period=[_period("2026-01", 2.0), _period("2026-02", -4.0)],
        )
        base.update(kw)
        return WalkForwardResult(**base)

    def test_derives_worst_fold_and_month(self):
        row = build_row(self._wf(), VerdictThresholds())
        self.assertEqual(row.worst_fold_return_pct, -5.0)
        self.assertEqual(row.worst_month_return_pct, -4.0)

    def test_instrument_contribution_string(self):
        row = build_row(self._wf(), VerdictThresholds())
        self.assertIn("EURUSD:+3.0%(30t)", row.instrument_contribution)
        self.assertIn("GBPUSD:+1.0%(20t)", row.instrument_contribution)

    def test_dominance_makes_it_fail(self):
        # EURUSD 3.0 of (3.0+1.0) = 75% > 70% default -> dominance failure.
        row = build_row(self._wf(), VerdictThresholds())
        self.assertFalse(row.passed)
        self.assertTrue(any("one instrument" in r for r in row.failed_rules))
        self.assertEqual(row.verdict, "FAIL")

    def test_balanced_book_passes(self):
        row = build_row(
            self._wf(by_instrument=[_instr("EURUSD", 2.0), _instr("GBPUSD", 2.0)]),
            VerdictThresholds(),
        )
        self.assertTrue(row.passed)
        self.assertEqual(row.failed_rules, [])
        self.assertEqual(row.verdict, "PASS")

    def test_empty_folds_and_periods_default_to_zero(self):
        row = build_row(self._wf(folds=[], by_period=[], by_instrument=[]), VerdictThresholds())
        self.assertEqual(row.worst_fold_return_pct, 0.0)
        self.assertEqual(row.worst_month_return_pct, 0.0)
        self.assertEqual(row.instrument_contribution, "")


class TestExport(unittest.TestCase):
    def _report(self) -> StrategyReport:
        rows = [
            StrategyRow("alpha", 5.0, 1.5, 70.0, 60, 0.3, -2.0, -3.0,
                        "EURUSD:+5.0%(60t)", True, []),
            StrategyRow("beta", -1.0, 0.8, 30.0, 40, -0.1, -8.0, -9.0,
                        "EURUSD:-1.0%(40t)", False, ["PF 0.80 < 1.10", "trades 40 < 40"]),
        ]
        meta = {"generated": "2026-06-28", "instruments": ["EURUSD"],
                "data_start": "2024-01-01", "data_end": "2024-03-01",
                "is_bars": 120, "oos_bars": 60, "step_bars": 60,
                "metric": "sharpe", "min_trades": 5, "n_strategies": 2}
        return StrategyReport(rows=rows, meta=meta)

    def test_csv_header_and_row_count(self):
        text = to_csv(self._report())
        parsed = list(csv.reader(io.StringIO(text)))
        self.assertEqual(parsed[0], COLUMNS)
        self.assertEqual(len(parsed), 3)  # header + 2 rows
        self.assertEqual(parsed[1][0], "alpha")
        self.assertEqual(parsed[1][-2], "PASS")
        # failed_rules with a comma-free " | " join survives CSV round-trip intact.
        self.assertEqual(parsed[2][-1], "PF 0.80 < 1.10 | trades 40 < 40")

    def test_markdown_has_header_table_and_verdict(self):
        md = to_markdown(self._report())
        self.assertIn("# Strategy rejection report", md)
        self.assertIn("data range: 2024-01-01 .. 2024-03-01", md)
        self.assertIn("| strategy |", md)
        self.assertIn("**PASS:** alpha", md)

    def test_markdown_no_pass(self):
        rep = self._report()
        for r in rep.rows:
            r.passed = False
        self.assertIn("No strategy passed", to_markdown(rep))


class TestRunStrategyReport(unittest.TestCase):
    def _config(self) -> TradingConfig:
        return TradingConfig(
            starting_equity=10_000.0,
            instruments=[InstrumentConfig(epic="TEST", timeframe="MINUTE_15")],
            risk=RiskConfig(),
            costs=CostConfig(spread_points=0.0),
        )

    def test_runs_all_strategies_and_sorts(self):
        # A gently trending series so trend strategies actually trade.
        closes = [1.10 + 0.0003 * i for i in range(320)]
        candles = {"TEST": make_candles(closes, epic="TEST")}
        config = self._config()
        grids = {
            "ema_crossover": {"fast": [5], "slow": [20]},
            "rsi_reversion": {"period": [14]},
        }
        wf_kwargs = dict(is_bars=120, oos_bars=60, step_bars=60,
                         metric="total_return", warmup_bars=30, min_trades=1,
                         select_top_n=None, select_metric="return")
        report = run_strategy_report(
            candles, config, list(grids), lambda n: grids.get(n, {}),
            wf_kwargs, VerdictThresholds(),
        )
        self.assertEqual({r.strategy for r in report.rows}, set(grids))
        # Rows sorted by OOS return descending.
        rets = [r.oos_return_pct for r in report.rows]
        self.assertEqual(rets, sorted(rets, reverse=True))
        # Provenance recorded.
        self.assertEqual(report.meta["instruments"], ["TEST"])
        self.assertEqual(report.meta["metric"], "total_return")
        self.assertTrue(report.meta["data_start"])

    def test_skips_strategy_without_grid(self):
        candles = {"TEST": make_candles([1.10 + 0.0003 * i for i in range(320)], epic="TEST")}
        skipped = []
        report = run_strategy_report(
            candles, self._config(), ["ema_crossover", "nogrid"],
            lambda n: {"fast": [5], "slow": [20]} if n == "ema_crossover" else {},
            dict(is_bars=120, oos_bars=60, step_bars=60, metric="total_return",
                 warmup_bars=30, min_trades=1, select_top_n=None, select_metric="return"),
            VerdictThresholds(),
            on_skip=lambda name, reason: skipped.append((name, reason)),
        )
        self.assertEqual([r.strategy for r in report.rows], ["ema_crossover"])
        self.assertEqual(skipped, [("nogrid", "no parameter grid")])


if __name__ == "__main__":
    unittest.main()

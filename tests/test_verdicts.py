import types
import unittest

from forex_bot.research.verdicts import (
    VerdictThresholds,
    evaluate,
    thresholds_from_config,
)


def _passing_kwargs(**overrides):
    """A baseline set of metrics that clears every rule; override one to fail it."""
    base = dict(
        oos_return_pct=5.0,
        profit_factor=1.5,
        pct_positive_folds=70.0,
        total_trades=60,
        avg_r_multiple=0.3,
        r_multiple_trades=60,
        worst_fold_return_pct=-3.0,
        worst_month_return_pct=-2.0,
        instrument_returns=[3.0, 2.0, 1.0],
    )
    base.update(overrides)
    return base


class TestVerdictRules(unittest.TestCase):
    def test_all_pass(self):
        v = evaluate(**_passing_kwargs())
        self.assertTrue(v.passed)
        self.assertEqual(v.failed_rules, [])

    def test_oos_return_at_floor_fails(self):
        v = evaluate(**_passing_kwargs(oos_return_pct=0.0))
        self.assertFalse(v.passed)
        self.assertTrue(any("OOS return" in r for r in v.failed_rules))

    def test_profit_factor_below_min_fails(self):
        v = evaluate(**_passing_kwargs(profit_factor=1.05))
        self.assertFalse(v.passed)
        self.assertTrue(any(r.startswith("PF") for r in v.failed_rules))

    def test_positive_folds_below_min_fails(self):
        v = evaluate(**_passing_kwargs(pct_positive_folds=40.0))
        self.assertFalse(v.passed)
        self.assertTrue(any("positive folds" in r for r in v.failed_rules))

    def test_too_few_trades_fails(self):
        v = evaluate(**_passing_kwargs(total_trades=20))
        self.assertFalse(v.passed)
        self.assertTrue(any(r.startswith("trades") for r in v.failed_rules))

    def test_nonpositive_avg_r_fails_when_defined(self):
        v = evaluate(**_passing_kwargs(avg_r_multiple=-0.1, r_multiple_trades=60))
        self.assertFalse(v.passed)
        self.assertTrue(any("avg R" in r for r in v.failed_rules))

    def test_avg_r_skipped_when_no_stops(self):
        # Negative avg R but no trade carried a stop -> rule is N/A, not a failure.
        v = evaluate(**_passing_kwargs(avg_r_multiple=-0.5, r_multiple_trades=0))
        self.assertTrue(v.passed)
        self.assertIn("avg R n/a (no stops)", v.notes)

    def test_instrument_dominance_fails(self):
        v = evaluate(**_passing_kwargs(instrument_returns=[10.0, 0.5, -1.0]))
        self.assertFalse(v.passed)
        self.assertTrue(any("one instrument" in r for r in v.failed_rules))

    def test_single_instrument_skips_dominance(self):
        v = evaluate(**_passing_kwargs(instrument_returns=[10.0]))
        self.assertTrue(v.passed)

    def test_breadth_too_few_positive_instruments_fails(self):
        # 1 of 3 instruments positive -> below the 50% breadth floor.
        v = evaluate(**_passing_kwargs(instrument_returns=[3.0, -1.0, -2.0]))
        self.assertFalse(v.passed)
        self.assertTrue(any("positive instruments" in r for r in v.failed_rules))

    def test_worst_fold_breach_fails(self):
        v = evaluate(**_passing_kwargs(worst_fold_return_pct=-20.0))
        self.assertFalse(v.passed)
        self.assertTrue(any("worst fold" in r for r in v.failed_rules))

    def test_worst_month_breach_fails(self):
        v = evaluate(**_passing_kwargs(worst_month_return_pct=-15.0))
        self.assertFalse(v.passed)
        self.assertTrue(any("worst month" in r for r in v.failed_rules))

    def test_multiple_failures_listed_together(self):
        v = evaluate(**_passing_kwargs(profit_factor=0.8, total_trades=5))
        self.assertFalse(v.passed)
        self.assertGreaterEqual(len(v.failed_rules), 2)


class TestThresholdsFromConfig(unittest.TestCase):
    def test_overrides_applied_and_unknown_ignored(self):
        cfg = types.SimpleNamespace(report={"min_trades": 10, "bogus_key": 99})
        t = thresholds_from_config(cfg)
        self.assertEqual(t.min_trades, 10)
        # Untouched fields keep their strict defaults.
        self.assertEqual(t.min_profit_factor, VerdictThresholds().min_profit_factor)

    def test_missing_report_yields_defaults(self):
        cfg = types.SimpleNamespace()  # no .report attribute
        self.assertEqual(thresholds_from_config(cfg), VerdictThresholds())


if __name__ == "__main__":
    unittest.main()

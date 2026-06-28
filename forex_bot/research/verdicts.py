"""Mechanical pass/fail rules for strategy rejection.

The point of this module is to turn "doesn't feel robust" into explicit,
deterministic rejection logic. It is intentionally pure — no I/O, no walk-forward
execution — so the same rules can be reused by the report writer
(``strategy_report.py``) and the inline CLI comparison without drifting apart.

A strategy PASSES only if it clears every active rule. Each rule that fails
records a short, human-readable reason string; the verdict is simply
``passed = not failed_rules``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class VerdictThresholds:
    """Rejection boundaries. Defaults are deliberately strict — the report exists
    to reject fast, not to flatter borderline strategies."""

    min_oos_return_pct: float = 0.0        # OOS return must be strictly greater
    min_profit_factor: float = 1.1
    min_pct_positive_folds: float = 50.0
    min_trades: int = 40
    min_avg_r: float = 0.0                 # avg R must be strictly greater (when defined)
    max_instrument_dominance: float = 0.70  # one instrument's share of positive return
    worst_fold_floor_pct: float = -10.0    # worst single OOS fold may not breach this
    worst_month_floor_pct: float = -10.0   # worst OOS month may not breach this


DEFAULT_THRESHOLDS = VerdictThresholds()


def thresholds_from_config(config: Any) -> VerdictThresholds:
    """Build thresholds from a ``report:`` config block, falling back to defaults.

    Unknown keys are ignored so a typo in config never silently weakens a rule;
    only the documented threshold fields are read.
    """
    raw = getattr(config, "report", None) or {}
    known = {f.name for f in fields(VerdictThresholds)}
    overrides = {k: raw[k] for k in known if k in raw}
    return VerdictThresholds(**overrides)


@dataclass
class Verdict:
    passed: bool
    failed_rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def evaluate(
    *,
    oos_return_pct: float,
    profit_factor: float,
    pct_positive_folds: float,
    total_trades: int,
    avg_r_multiple: float,
    r_multiple_trades: int,
    worst_fold_return_pct: float,
    worst_month_return_pct: float,
    instrument_returns: Sequence[float],
    thresholds: VerdictThresholds = DEFAULT_THRESHOLDS,
) -> Verdict:
    """Apply every rule and return a pass/fail verdict with reasons."""
    t = thresholds
    failed: list[str] = []
    notes: list[str] = []

    if oos_return_pct <= t.min_oos_return_pct:
        failed.append(f"OOS return {oos_return_pct:.2f}% <= {t.min_oos_return_pct:.2f}%")
    if profit_factor < t.min_profit_factor:
        failed.append(f"PF {profit_factor:.2f} < {t.min_profit_factor:.2f}")
    if pct_positive_folds < t.min_pct_positive_folds:
        failed.append(
            f"positive folds {pct_positive_folds:.0f}% < {t.min_pct_positive_folds:.0f}%"
        )
    if total_trades < t.min_trades:
        failed.append(f"trades {total_trades} < {t.min_trades}")

    # Avg R is only meaningful when at least one trade carried a stop. With no
    # stops, R is undefined for every trade — skip the rule rather than reject a
    # strategy for a metric it cannot produce.
    if r_multiple_trades > 0:
        if avg_r_multiple <= t.min_avg_r:
            failed.append(f"avg R {avg_r_multiple:.2f} <= {t.min_avg_r:.2f}")
    else:
        notes.append("avg R n/a (no stops)")

    # Concentration: reject when a single instrument supplies most of the positive
    # return while the basket as a whole leans on it. Only checked with >1
    # instrument and some positive contribution to divide by.
    positives = [r for r in instrument_returns if r > 0]
    if len(instrument_returns) > 1 and positives:
        total_pos = sum(positives)
        share = max(positives) / total_pos if total_pos > 0 else 0.0
        if share > t.max_instrument_dominance:
            failed.append(
                f"one instrument = {share:.0%} of positive return "
                f"> {t.max_instrument_dominance:.0%}"
            )

    if worst_fold_return_pct < t.worst_fold_floor_pct:
        failed.append(
            f"worst fold {worst_fold_return_pct:.2f}% < {t.worst_fold_floor_pct:.2f}%"
        )
    if worst_month_return_pct < t.worst_month_floor_pct:
        failed.append(
            f"worst month {worst_month_return_pct:.2f}% < {t.worst_month_floor_pct:.2f}%"
        )

    return Verdict(passed=not failed, failed_rules=failed, notes=notes)

"""Research tooling: walk-forward optimization, OOS evaluation, pair screening."""

from .registry import (
    FROZEN_STATUSES,
    STATUSES,
    RegistryEntry,
    ResearchRegistry,
    setup_key,
)
from .screening import PairStat, pair_stat, screen_pairs, screen_report
from .strategy_report import (
    StrategyReport,
    StrategyRow,
    build_row,
    run_strategy_report,
    to_csv,
    to_markdown,
)
from .verdicts import (
    Verdict,
    VerdictThresholds,
    classify_holdout,
    evaluate,
    thresholds_from_config,
)
from .walkforward import (
    Fold,
    HoldoutResult,
    WalkForwardResult,
    grid_search,
    holdout_test,
    param_combinations,
    walk_forward,
)

__all__ = [
    "walk_forward",
    "holdout_test",
    "grid_search",
    "param_combinations",
    "Fold",
    "WalkForwardResult",
    "HoldoutResult",
    "screen_pairs",
    "screen_report",
    "pair_stat",
    "PairStat",
    "run_strategy_report",
    "build_row",
    "to_csv",
    "to_markdown",
    "StrategyReport",
    "StrategyRow",
    "VerdictThresholds",
    "Verdict",
    "evaluate",
    "classify_holdout",
    "thresholds_from_config",
    "ResearchRegistry",
    "RegistryEntry",
    "setup_key",
    "STATUSES",
    "FROZEN_STATUSES",
]

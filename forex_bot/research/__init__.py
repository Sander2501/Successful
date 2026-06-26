"""Research tooling: walk-forward optimization, OOS evaluation, pair screening."""

from .screening import PairStat, pair_stat, screen_pairs, screen_report
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
]

"""Research tooling: walk-forward optimization, OOS evaluation, pair screening."""

from .screening import PairStat, pair_stat, screen_pairs, screen_report
from .walkforward import (
    Fold,
    WalkForwardResult,
    grid_search,
    param_combinations,
    walk_forward,
)

__all__ = [
    "walk_forward",
    "grid_search",
    "param_combinations",
    "Fold",
    "WalkForwardResult",
    "screen_pairs",
    "screen_report",
    "pair_stat",
    "PairStat",
]

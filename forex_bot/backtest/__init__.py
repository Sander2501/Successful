"""Backtesting & simulation layer."""

from .engine import Backtester, BacktestResult
from .metrics import PerformanceReport, compute_metrics
from .portfolio import Portfolio

__all__ = [
    "Backtester",
    "BacktestResult",
    "Portfolio",
    "compute_metrics",
    "PerformanceReport",
]

"""Pair screening: find spreads worth trading *before* backtesting them.

Spread/pairs strategies only work when the spread is genuinely mean-reverting.
Rather than trying every pair and keeping whatever looked good (data-snooping),
this screens candidates by a statistical property — the spread's mean-reversion
half-life — so the walk-forward only has to judge a principled shortlist.

For each pair it fits a hedge ratio (OLS of log prices), builds the spread, and
estimates the Ornstein-Uhlenbeck half-life from the AR(1) coefficient of the
spread. A short, finite half-life (roughly a handful to a few hundred bars) means
the spread reverts at a tradeable speed; no reversion (or a half-life longer than
the data) means it is effectively a random walk and not worth trading.

Pure stdlib. Screening is necessary but not sufficient: a mean-reverting spread
can still be untradeable after costs — that is what the backtest/optimize step
decides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import Candle

# Engle-Granger ADF critical value (~5%, one regressor, no trend). The spread is
# residual from a cointegrating regression, so the usual DF value (-2.86) is too
# lenient; this stricter bound rejects most spurious-regression false positives.
ADF_CRITICAL_5PCT = -3.34
# Cointegration alone is fooled by finite samples; genuinely linked pairs also
# co-move, so require a minimum absolute return correlation as a second filter.
MIN_ABS_CORRELATION = 0.3


@dataclass
class PairStat:
    epic_a: str
    epic_b: str
    correlation: float
    beta: float
    half_life: float | None  # bars; None if not mean-reverting
    adf_t: float                # Engle-Granger ADF t-stat on the spread
    n: int

    @property
    def cointegrated(self) -> bool:
        """Spread rejects a unit root (stationary) at ~5%."""
        return self.adf_t < ADF_CRITICAL_5PCT

    @property
    def mean_reverting(self) -> bool:
        return self.cointegrated and self.half_life is not None

    @property
    def tradeable_speed(self) -> bool:
        """A real candidate: cointegrated, co-moving, and reverting at a tradeable
        speed. The correlation floor rejects spurious cointegration with an
        unrelated series."""
        return (
            self.mean_reverting
            and abs(self.correlation) >= MIN_ABS_CORRELATION
            and 3.0 <= self.half_life <= 250.0
        )


def _aligned_log_closes(
    a: list[Candle], b: list[Candle]
) -> tuple[list[float], list[float]]:
    ma = {c.timestamp: c.close for c in a}
    mb = {c.timestamp: c.close for c in b}
    common = sorted(set(ma) & set(mb))
    la = [math.log(ma[t]) for t in common if ma[t] > 0 and mb[t] > 0]
    lb = [math.log(mb[t]) for t in common if ma[t] > 0 and mb[t] > 0]
    return la, lb


def _ols(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (intercept, slope) for y ~ a + b*x."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    b = sxy / sxx
    return my - b * mx, b


def _correlation(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    dx = [x[i + 1] - x[i] for i in range(n - 1)]
    dy = [y[i + 1] - y[i] for i in range(n - 1)]
    mx = sum(dx) / len(dx)
    my = sum(dy) / len(dy)
    vx = sum((d - mx) ** 2 for d in dx)
    vy = sum((d - my) ** 2 for d in dy)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((dx[i] - mx) * (dy[i] - my) for i in range(len(dx)))
    return cov / math.sqrt(vx * vy)


def _spread_dynamics(spread: list[float]) -> tuple[float | None, float]:
    """Return (half_life, adf_t) for the spread.

    Fits the Dickey-Fuller regression Δs_t = a + k·s_{t-1} + e, then derives the
    OU half-life from k and the ADF t-statistic of k (its distance below zero,
    in standard errors). A strongly negative t-stat means the spread reverts and
    is unlikely to be a spurious fit.
    """
    if len(spread) < 10:
        return None, 0.0
    lag = spread[:-1]
    delta = [spread[i] - spread[i - 1] for i in range(1, len(spread))]
    n = len(lag)
    a, k = _ols(lag, delta)
    # t-stat of k: k / se(k), se(k) = sqrt(sigma^2 / Sxx), sigma^2 = SSR/(n-2).
    mx = sum(lag) / n
    sxx = sum((x - mx) ** 2 for x in lag)
    if sxx <= 0 or n <= 2:
        return None, 0.0
    ssr = sum((delta[i] - (a + k * lag[i])) ** 2 for i in range(n))
    sigma2 = ssr / (n - 2)
    se_k = math.sqrt(sigma2 / sxx) if sigma2 > 0 else 0.0
    adf_t = (k / se_k) if se_k > 0 else 0.0

    phi = 1.0 + k
    half_life = None
    if 0.0 < phi < 1.0:
        hl = -math.log(2) / math.log(phi)
        half_life = hl if hl > 0 else None
    return half_life, adf_t


def pair_stat(epic_a: str, a: list[Candle], epic_b: str, b: list[Candle]) -> PairStat:
    la, lb = _aligned_log_closes(a, b)
    n = min(len(la), len(lb))
    if n < 30:
        return PairStat(epic_a, epic_b, 0.0, 0.0, None, 0.0, n)
    _intercept, beta = _ols(lb, la)  # la ~ beta*lb
    spread = [la[i] - beta * lb[i] for i in range(n)]
    half_life, adf_t = _spread_dynamics(spread)
    return PairStat(
        epic_a=epic_a,
        epic_b=epic_b,
        correlation=_correlation(la, lb),
        beta=beta,
        half_life=half_life,
        adf_t=adf_t,
        n=n,
    )


def screen_pairs(candles_by_epic: dict[str, list[Candle]]) -> list[PairStat]:
    """Stats for every instrument pair, best mean-reversion candidates first."""
    epics = sorted(candles_by_epic)
    stats: list[PairStat] = []
    for i, a in enumerate(epics):
        for b in epics[i + 1 :]:
            stats.append(pair_stat(a, candles_by_epic[a], b, candles_by_epic[b]))

    def rank_key(s: PairStat):
        # Tradeable cointegrated pairs first, then by ADF strength (more
        # negative = more confidently stationary), then by half-life.
        hl = s.half_life if s.half_life is not None else float("inf")
        return (0 if s.tradeable_speed else 1, s.adf_t, hl)

    return sorted(stats, key=rank_key)


def screen_report(stats: list[PairStat]) -> str:
    lines = [
        "Pair screen (cointegration candidates for spread_reversion):",
        f"  {'pair':<20} {'corr':>6} {'beta':>7} {'ADF_t':>7} {'half_life':>10} {'verdict':>14}",
    ]
    for s in stats:
        hl = f"{s.half_life:.1f}" if s.half_life is not None else "—"
        if s.tradeable_speed:
            verdict = "TEST IT"
        elif s.cointegrated:
            verdict = "slow/fast"
        else:
            verdict = "not cointegr."
        lines.append(
            f"  {s.epic_a + '/' + s.epic_b:<20} {s.correlation:>6.2f} "
            f"{s.beta:>7.2f} {s.adf_t:>7.2f} {hl:>10} {verdict:>14}"
        )
    candidates = [s for s in stats if s.tradeable_speed]
    lines.append("")
    if candidates:
        best = candidates[0]
        lines.append(
            f"Best candidate: {best.epic_a}/{best.epic_b} "
            f"(half-life ~{best.half_life:.0f} bars). Set these two as your "
            f"instruments and run: optimize --all-strategies"
        )
        lines.append("NB: a mean-reverting spread is necessary, not sufficient — the "
                     "out-of-sample backtest still has to clear costs.")
    else:
        lines.append("No pair has a tradeable-speed mean-reverting spread here. "
                     "Spread trading is unlikely to work on this set; try "
                     "economically-linked crosses (EURGBP, AUDNZD, EURCHF).")
    return "\n".join(lines)

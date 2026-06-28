"""Smart-Money-Concepts (SMC) sweep-reversal strategy.

This is a deliberately *mechanical* and *small* interpretation of SMC. Every term
that practitioners usually eyeball ("liquidity", "change of character", "fair
value gap") is given an exact, falsifiable definition so the strategy can be
backtested, walk-forwarded and held out like Donchian or RSI — not narrated in
hindsight.

Version 1 setup (and only this):

    liquidity sweep  ->  CHOCH (change of character)  ->  FVG/OB retrace entry

Definitions (all computed on CLOSED candles only — no look-ahead):

* Swing point (fractal of strength ``k``): a high that strictly exceeds the
  ``k`` highs on each side (mirror for lows). A swing is only "confirmed" once
  ``k`` candles have printed after it, so a swing at index ``i`` is never used
  before bar ``i + k`` exists.

* Liquidity sweep: a candle that trades *through* a prior confirmed swing
  (grabbing the stops resting beyond it) but *closes back* on the original side
  — a failed breakout. Sweep of a swing high = bearish intent; sweep of a swing
  low = bullish intent.

* CHOCH (change of character): after a sweep, structure flips when price closes
  beyond the opposing confirmed swing (below the protected swing low for a short,
  above the protected swing high for a long). This is the displacement that
  confirms the reversal.

* FVG (fair value gap): a 3-candle imbalance inside the displacement leg. Bearish
  FVG = ``low[i-2] > high[i]`` (zone ``[high[i], low[i-2]]``); bullish FVG =
  ``high[i-2] < low[i]`` (zone ``[high[i-2], low[i]]``). The entry triggers when
  price retraces *into* that zone. If no qualifying FVG exists, the order block
  (last opposite-colour candle before the displacement) is used as the zone.

* Higher-timeframe (HTF) liquidity for targets: the LTF stream is aggregated into
  HTF candles in fixed blocks of ``htf_factor`` *completed* LTF bars, and the
  nearest opposing HTF swing becomes the profit target. No second data feed is
  used, so there is no cross-timeframe look-ahead.

Stops are structural (just beyond the sweep wick), not a bare ATR multiple, which
is the main reason this can carry a better reward:risk than trend-following.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..indicators import atr
from ..models import Candle, Signal, SignalType
from .base import StrategyBase, StrategyContext


@dataclass
class SmcAnalysis:
    """Structured, stateless read of the current market for one instrument.

    ``direction`` is ``"long"``, ``"short"`` or ``None`` (no actionable setup).
    When a setup is live, ``entry_zone`` is the (low, high) retrace band, and a
    signal should fire only while price is inside it. ``reasons`` records *why*
    (which sweep level, CHOCH level, zone type) for per-trade auditing so setup
    quality can be sliced after the fact instead of trusted blind.
    """

    direction: str | None = None
    entry_zone: tuple[float, float] | None = None
    stop: float | None = None
    target: float | None = None
    reasons: dict = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return (
            self.direction in ("long", "short")
            and self.entry_zone is not None
            and self.stop is not None
        )


def _confirmed_swings(
    highs: Sequence[float], lows: Sequence[float], k: int
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return ``(swing_highs, swing_lows)`` as ``(index, price)`` lists.

    Only fractals with ``k`` candles on *both* sides are returned, so every
    swing here is already confirmed and safe to reference without look-ahead.
    """
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    n = len(highs)
    for i in range(k, n - k):
        h = highs[i]
        if all(h > highs[i - j] for j in range(1, k + 1)) and all(
            h > highs[i + j] for j in range(1, k + 1)
        ):
            swing_highs.append((i, h))
        lo = lows[i]
        if all(lo < lows[i - j] for j in range(1, k + 1)) and all(
            lo < lows[i + j] for j in range(1, k + 1)
        ):
            swing_lows.append((i, lo))
    return swing_highs, swing_lows


def _htf_swing_target(
    candles: Sequence[Candle], htf_factor: int, k: int, direction: str, entry: float
) -> float | None:
    """Nearest opposing HTF swing as a liquidity target.

    The LTF stream is collapsed into non-overlapping blocks of ``htf_factor``
    completed candles; a partial trailing block is dropped so only closed HTF
    bars are considered. For a short we look for the nearest HTF swing low below
    ``entry`` (resting liquidity to draw toward); mirror for a long.
    """
    if htf_factor < 2:
        return None
    blocks = len(candles) // htf_factor
    if blocks < 2 * k + 1:
        return None
    htf_high: list[float] = []
    htf_low: list[float] = []
    for b in range(blocks):
        chunk = candles[b * htf_factor : (b + 1) * htf_factor]
        htf_high.append(max(c.high for c in chunk))
        htf_low.append(min(c.low for c in chunk))
    sh, sl = _confirmed_swings(htf_high, htf_low, k)
    if direction == "short":
        below = [p for _, p in sl if p < entry]
        return max(below) if below else None
    above = [p for _, p in sh if p > entry]
    return min(above) if above else None


class SmcMarketAnalyzer:
    """Pure, stateless detector: candles in, :class:`SmcAnalysis` out.

    Holds no state between calls — every analysis is recomputed from the supplied
    history — so it is trivially testable and identical in backtest and live.
    """

    def __init__(
        self,
        *,
        swing_k: int = 2,
        lookback: int = 60,
        htf_factor: int = 4,
        atr_period: int = 14,
        stop_buffer_atr: float = 0.1,
        min_rr: float = 2.5,
        tp_rr: float = 3.0,
        min_pen_atr: float = 0.0,
        min_rej_frac: float = 0.0,
        fvg_prefer: str = "sweep",
    ) -> None:
        if swing_k < 1:
            raise ValueError("swing_k must be >= 1")
        if lookback < swing_k * 4:
            raise ValueError("lookback too small for the chosen swing_k")
        if fvg_prefer not in ("sweep", "choch"):
            raise ValueError("fvg_prefer must be 'sweep' or 'choch'")
        self.swing_k = swing_k
        self.lookback = lookback
        self.htf_factor = htf_factor
        self.atr_period = atr_period
        self.stop_buffer_atr = stop_buffer_atr
        # min_rr defaults high on purpose: a sweep-reversal wins a minority of the
        # time, so breakeven RR = (1-WR)/WR is large (≈2.8 at a 26% win rate). A
        # low min_rr is a structural loser regardless of detection quality.
        self.min_rr = min_rr
        self.tp_rr = tp_rr
        # Sweep-quality gate (off at 0.0): a real stop-grab must PENETRATE the
        # swept level by min_pen_atr*ATR and then CLOSE back at least min_rej_frac
        # of that penetration inside — a decisive rejection, not a drift-through.
        self.min_pen_atr = min_pen_atr
        self.min_rej_frac = min_rej_frac
        # Which FVG in the displacement leg to use for the retrace entry. "sweep"
        # (default) takes the gap nearest the sweep => better entry price, tighter
        # stop, higher RR (fills less often). "choch" takes the gap nearest the
        # break => fills more often, worse RR.
        self.fvg_prefer = fvg_prefer

    # ------------------------------------------------------------------ #
    def analyze(self, candles: Sequence[Candle]) -> SmcAnalysis:
        n = len(candles)
        if n < self.lookback:
            return SmcAnalysis()

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        cur = n - 1

        swing_highs, swing_lows = _confirmed_swings(highs, lows, self.swing_k)
        if not swing_highs or not swing_lows:
            return SmcAnalysis()

        atr_series = atr(highs, lows, closes, self.atr_period)
        atr_now = atr_series[-1] if atr_series and atr_series[-1] is not None else None
        if not atr_now:
            return SmcAnalysis()

        window_start = max(0, cur - self.lookback)

        short = self._detect(
            candles, highs, lows, closes, swing_highs, swing_lows,
            atr_now, window_start, cur, direction="short",
        )
        if short.actionable:
            return short
        return self._detect(
            candles, highs, lows, closes, swing_lows, swing_highs,
            atr_now, window_start, cur, direction="long",
        )

    # ------------------------------------------------------------------ #
    def _detect(
        self, candles, highs, lows, closes, swept_swings, choch_swings,
        atr_now, window_start, cur, *, direction: str,
    ) -> SmcAnalysis:
        """Generic detector; ``direction`` selects the mirror of the same logic.

        For ``short``: sweep of a swing HIGH, CHOCH below a swing LOW, bearish
        FVG, entry on a retrace UP into the zone, stop ABOVE the sweep wick.
        For ``long`` the caller passes the swing lists swapped and the sign
        comparisons below flip via ``s``.
        """
        s = 1.0 if direction == "short" else -1.0

        # 1. Most recent liquidity sweep inside the lookback window.
        sweep = self._find_sweep(highs, lows, closes, swept_swings,
                                 window_start, cur, direction, atr_now)
        if sweep is None:
            return SmcAnalysis()
        sweep_bar, sweep_extreme, swept_level = sweep

        # 2. CHOCH: close beyond the opposing confirmed swing, after the sweep.
        choch = self._find_choch(closes, choch_swings, sweep_bar, cur, direction)
        if choch is None:
            return SmcAnalysis()
        choch_bar, choch_level = choch

        # 3. Retrace zone: a qualifying FVG in the displacement leg, else the OB.
        zone = self._displacement_zone(candles, sweep_bar, choch_bar, direction)
        if zone is None:
            return SmcAnalysis()
        zone_low, zone_high, zone_kind = zone

        # 4. Entry trigger: the current candle taps into the zone, and price has
        #    not already run past the swept level (setup still valid).
        cur_low, cur_high, cur_close = lows[cur], highs[cur], closes[cur]
        tapped = cur_low <= zone_high and cur_high >= zone_low
        if not tapped:
            return SmcAnalysis()
        if direction == "short" and cur_close > swept_level:
            return SmcAnalysis()  # broke back above swept high -> invalid
        if direction == "long" and cur_close < swept_level:
            return SmcAnalysis()

        # 5. Structural stop just beyond the sweep wick.
        buffer = self.stop_buffer_atr * atr_now
        stop = sweep_extreme + s * buffer
        entry = cur_close
        risk = abs(entry - stop)
        if risk <= 0:
            return SmcAnalysis()

        # 6. Target: nearest opposing HTF swing liquidity, else an R-multiple.
        htf = _htf_swing_target(candles[: cur + 1], self.htf_factor,
                                self.swing_k, direction, entry)
        rr_target = entry - s * self.tp_rr * risk
        target = htf if htf is not None else rr_target
        # Never accept a target on the wrong side of entry.
        if (direction == "short" and target >= entry) or (
            direction == "long" and target <= entry
        ):
            target = rr_target
        rr = abs(target - entry) / risk
        if rr < self.min_rr:
            return SmcAnalysis()

        return SmcAnalysis(
            direction=direction,
            entry_zone=(zone_low, zone_high),
            stop=stop,
            target=target,
            reasons={
                "swept_level": round(swept_level, 6),
                "sweep_bar_ago": cur - sweep_bar,
                "choch_level": round(choch_level, 6),
                "zone": zone_kind,
                "target_src": "htf_swing" if htf is not None else "rr",
                "rr": round(rr, 2),
            },
        )

    # ------------------------------------------------------------------ #
    def _find_sweep(self, highs, lows, closes, swings, window_start, cur, direction, atr_now):
        """Latest *quality* sweep in the window: pokes past the most recent
        confirmed swing then closes back on the original side (a failed
        breakout), with an optional penetration/rejection gate."""
        for j in range(cur, window_start - 1, -1):
            prior = [(i, p) for i, p in swings if i + self.swing_k <= j and i < j]
            if not prior:
                continue
            _, level = prior[-1]  # most recent confirmed swing
            if direction == "short":
                if highs[j] > level and closes[j] < level:
                    if self._sweep_quality_ok(highs[j] - level, level - closes[j], atr_now):
                        return j, highs[j], level
            else:
                if lows[j] < level and closes[j] > level:
                    if self._sweep_quality_ok(level - lows[j], closes[j] - level, atr_now):
                        return j, lows[j], level
        return None

    def _sweep_quality_ok(self, penetration: float, rejection: float, atr_now: float) -> bool:
        """A quality stop-grab penetrates the level decisively and closes back in."""
        if self.min_pen_atr > 0 and penetration < self.min_pen_atr * atr_now:
            return False
        if self.min_rej_frac > 0 and rejection < self.min_rej_frac * penetration:
            return False
        return True

    def _find_choch(self, closes, swings, sweep_bar, cur, direction):
        """First close beyond the most recent opposing swing (the protected
        structure level) after the sweep — the displacement that flips structure."""
        for m in range(sweep_bar + 1, cur + 1):
            prior = [(i, p) for i, p in swings if i + self.swing_k <= m and i <= sweep_bar]
            if not prior:
                continue
            _, level = prior[-1]  # most recent protected swing before the sweep
            if direction == "short":
                if closes[m] < level:
                    return m, level
            else:
                if closes[m] > level:
                    return m, level
        return None

    def _displacement_zone(self, candles, sweep_bar, choch_bar, direction):
        """Return ``(low, high, kind)`` for the retrace zone within the leg.

        Picks a qualifying FVG per ``fvg_prefer`` ("sweep" => the gap nearest the
        sweep, scanning forward, for the best entry geometry; "choch" => nearest
        the break, scanning backward, for higher fill probability). Falls back to
        the order block (last opposite-colour candle before the displacement).
        """
        start = max(sweep_bar + 2, 2)
        order = (range(start, choch_bar + 1) if self.fvg_prefer == "sweep"
                 else range(choch_bar, start - 1, -1))
        for i in order:
            a, c = candles[i - 2], candles[i]
            if direction == "short" and a.low > c.high:
                return c.high, a.low, "fvg"
            if direction == "long" and a.high < c.low:
                return a.high, c.low, "fvg"
        # Order block: last opposite-colour candle before the CHOCH close.
        for i in range(choch_bar, sweep_bar - 1, -1):
            ob = candles[i]
            if direction == "short" and ob.close > ob.open:  # last up candle
                return min(ob.open, ob.close), max(ob.open, ob.close), "ob"
            if direction == "long" and ob.close < ob.open:  # last down candle
                return min(ob.open, ob.close), max(ob.open, ob.close), "ob"
        return None


class SmcSweepReversalStrategy(StrategyBase):
    """One strategy wrapping :class:`SmcMarketAnalyzer`: the sweep-reversal setup.

    Entries are structural (stop beyond the sweep, target at HTF liquidity);
    exits are left to the protective SL/TP carried on the signal, plus an
    opposite-direction CHOCH which closes early if structure flips against us.
    """

    def __init__(
        self,
        swing_k: int = 2,
        lookback: int = 60,
        htf_factor: int = 4,
        atr_period: int = 14,
        stop_buffer_atr: float = 0.1,
        min_rr: float = 2.5,
        tp_rr: float = 3.0,
        min_pen_atr: float = 0.0,
        min_rej_frac: float = 0.0,
        fvg_prefer: str = "sweep",
        session_start_hour: int | None = None,
        session_end_hour: int | None = None,
    ) -> None:
        self.analyzer = SmcMarketAnalyzer(
            swing_k=swing_k,
            lookback=lookback,
            htf_factor=htf_factor,
            atr_period=atr_period,
            stop_buffer_atr=stop_buffer_atr,
            min_rr=min_rr,
            tp_rr=tp_rr,
            min_pen_atr=min_pen_atr,
            min_rej_frac=min_rej_frac,
            fvg_prefer=fvg_prefer,
        )
        # Optional liquid-hours gate (UTC). Both None => trade any hour. Set e.g.
        # 7..16 to confine entries to the London/NY window and skip the thin,
        # noisy Asia/rollover setups that pad the trade count with junk.
        self.session_start_hour = session_start_hour
        self.session_end_hour = session_end_hour
        self.warmup = max(lookback, htf_factor * (2 * swing_k + 1), atr_period) + 2

    def _in_session(self, candle: Candle) -> bool:
        if self.session_start_hour is None or self.session_end_hour is None:
            return True
        return self.session_start_hour <= candle.timestamp.hour < self.session_end_hour

    def on_candle(self, candle: Candle, context: StrategyContext) -> Signal | None:
        if len(context.history) < self.warmup:
            return None
        analysis = self.analyzer.analyze(context.history)
        position = context.position

        if position is not None:
            # Exit early only if structure flips against the open position.
            if analysis.direction == "short" and position.side.value == "BUY":
                return Signal(candle.epic, SignalType.EXIT, candle.timestamp,
                              meta={"reason": "choch_against", **analysis.reasons})
            if analysis.direction == "long" and position.side.value == "SELL":
                return Signal(candle.epic, SignalType.EXIT, candle.timestamp,
                              meta={"reason": "choch_against", **analysis.reasons})
            return None

        if not analysis.actionable:
            return None

        # Session gate applies to ENTRIES only; exits above always fire.
        if not self._in_session(candle):
            return None

        sig_type = (
            SignalType.ENTER_SHORT if analysis.direction == "short"
            else SignalType.ENTER_LONG
        )
        return Signal(
            candle.epic,
            sig_type,
            candle.timestamp,
            stop_loss=analysis.stop,
            take_profit=analysis.target,
            meta={"reason": "sweep_reversal", **analysis.reasons},
        )

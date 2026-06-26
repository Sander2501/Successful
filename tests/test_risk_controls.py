import unittest
from datetime import date

from forex_bot.backtest.engine import Backtester
from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.models import Signal, SignalType
from forex_bot.risk.manager import RiskManager
from forex_bot.strategy.base import StrategyBase
from tests.helpers import make_candles


class _AlwaysLong(StrategyBase):
    """Test stub: open and hold a long, never exit (no protective stop)."""

    warmup = 2

    def on_candle(self, candle, context):
        if context.position is None and len(context.closes) >= 2:
            return Signal(candle.epic, SignalType.ENTER_LONG, candle.timestamp)
        return None


class TestKillSwitch(unittest.TestCase):
    def test_trips_on_total_drawdown(self):
        cfg = RiskConfig(max_total_drawdown_pct=0.15, max_daily_loss_pct=1.0)
        rm = RiskManager(cfg)
        rm.start_day(date(2024, 1, 1), 10000)
        rm.update_equity(date(2024, 1, 1), 11000)  # new peak
        rm.update_equity(date(2024, 1, 2), 10000)  # -9% from peak: ok
        self.assertFalse(rm.killed)
        rm.update_equity(date(2024, 1, 3), 9200)   # -16% from 11000 peak: kill
        self.assertTrue(rm.killed)
        self.assertTrue(rm.halted)
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.99)
        dec = rm.evaluate(sig, price=1.0, equity=9200, positions=[])
        self.assertFalse(dec.approved)
        self.assertIn("kill switch", dec.reason)

    def test_manual_kill(self):
        rm = RiskManager(RiskConfig())
        self.assertFalse(rm.killed)
        rm.kill()
        self.assertTrue(rm.killed)

    def test_alltime_peak_is_sticky(self):
        # Default (no window): once tripped it stays killed even after recovery.
        rm = RiskManager(RiskConfig(max_total_drawdown_pct=0.20))
        self.assertFalse(rm.recoverable)
        d = date(2024, 1, 1)
        rm.start_day(d, 100.0)
        rm.update_equity(d, 100.0)
        rm.update_equity(d, 75.0)   # -25% -> kill
        self.assertTrue(rm.killed)
        rm.update_equity(d, 100.0)  # fully recovered, but sticky
        self.assertTrue(rm.killed)

    def test_rolling_peak_window_recovers(self):
        # With a window the peak expires and the switch re-arms (hysteresis at
        # half the limit) so the bot can resume on its own.
        rm = RiskManager(RiskConfig(max_total_drawdown_pct=0.20,
                                    drawdown_peak_window_bars=3))
        self.assertTrue(rm.recoverable)
        d = date(2024, 1, 1)
        rm.start_day(d, 100.0)
        rm.update_equity(d, 100.0)  # window [100]
        rm.update_equity(d, 120.0)  # [100,120]
        rm.update_equity(d, 90.0)   # [100,120,90] peak120 dd25% -> kill
        self.assertTrue(rm.killed)
        rm.update_equity(d, 100.0)  # [120,90,100] peak120 dd16.7% > 10% -> still killed
        self.assertTrue(rm.killed)
        rm.update_equity(d, 110.0)  # [90,100,110] peak110 dd0 <= 10% -> resume
        self.assertFalse(rm.killed)

    def test_engine_flattens_and_halts_on_kill(self):
        # Rise, then a sustained crash while holding a 1x long -> drawdown trips
        # the kill switch; the engine must flatten and stop opening.
        closes = [1.20 + 0.001 * i for i in range(20)]
        closes += [closes[-1] * (0.99 ** i) for i in range(1, 40)]  # ~33% crash
        candles = make_candles(closes, epic="EURUSD")
        cfg = TradingConfig(
            starting_equity=10000.0,
            instruments=[InstrumentConfig("EURUSD")],
            risk=RiskConfig(max_position_pct=1.0, max_open_positions=1,
                            max_daily_loss_pct=1.0, max_total_drawdown_pct=0.10),
            costs=CostConfig(spread_points=0.0),
        )
        bt = Backtester(_AlwaysLong(), cfg)
        res = bt.run({"EURUSD": candles})
        self.assertTrue(bt.risk.killed)
        # A kill-switch close must appear and no trade may start after the kill.
        self.assertTrue(bt._kill_flattened)


class TestVolTargetSizing(unittest.TestCase):
    def test_vol_target_inverse_to_atr(self):
        cfg = RiskConfig(sizing_mode="vol_target", vol_target_pct=0.01,
                         max_position_pct=1.0)
        rm = RiskManager(cfg, value_per_point=1.0)
        # vol_target: size = (0.01 * 10000) / atr. atr=0.001 -> 100000, but capped
        # by max_position_pct (1.0*10000/price=10000). Use atr large enough to be
        # under the cap to test the formula directly.
        sig = Signal("E", SignalType.ENTER_LONG, meta={"atr": 0.05})
        dec = rm.evaluate(sig, price=1.0, equity=10000, positions=[])
        # 100 / 0.05 = 2000
        self.assertAlmostEqual(dec.order.size, 2000.0)

    def test_equalizes_risk_across_volatility(self):
        cfg = RiskConfig(sizing_mode="vol_target", vol_target_pct=0.01,
                         max_position_pct=1.0)
        rm = RiskManager(cfg, value_per_point=1.0)
        low_vol = rm.evaluate(Signal("A", SignalType.ENTER_LONG, meta={"atr": 0.02}),
                              price=1.0, equity=10000, positions=[])
        high_vol = rm.evaluate(Signal("B", SignalType.ENTER_LONG, meta={"atr": 0.08}),
                               price=1.0, equity=10000, positions=[])
        # A 1-ATR move should cost the same on both (vol_target_pct * equity).
        risk_a = low_vol.order.size * 0.02
        risk_b = high_vol.order.size * 0.08
        self.assertAlmostEqual(risk_a, risk_b)
        self.assertAlmostEqual(risk_a, 100.0)  # 1% of 10000

    def test_falls_back_without_atr(self):
        cfg = RiskConfig(sizing_mode="vol_target", max_position_pct=0.5)
        rm = RiskManager(cfg, value_per_point=1.0)
        # No atr in meta -> fall back to stop-distance fixed-fractional sizing.
        sig = Signal("E", SignalType.ENTER_LONG, stop_loss=0.90)
        dec = rm.evaluate(sig, price=1.0, equity=10000, positions=[])
        self.assertTrue(dec.approved)
        self.assertAlmostEqual(dec.order.size, 100.0 / 0.10)  # risk_per_trade default 1%


if __name__ == "__main__":
    unittest.main()

import argparse
import tempfile
import unittest
from pathlib import Path

from forex_bot.cli import _instruments_from_args
from forex_bot.config import InstrumentConfig, TradingConfig


def _config():
    return TradingConfig(instruments=[InstrumentConfig("EURUSD", "HOUR_4")])


class TestInstrumentsFromArgs(unittest.TestCase):
    def test_epics_override(self):
        args = argparse.Namespace(epics="EURGBP, EURCHF ,AUDNZD", timeframe=None)
        insts = _instruments_from_args(_config(), args)
        self.assertEqual([i.epic for i in insts], ["EURGBP", "EURCHF", "AUDNZD"])
        # Timeframe falls back to the config's instrument timeframe.
        self.assertTrue(all(i.timeframe == "HOUR_4" for i in insts))

    def test_timeframe_override(self):
        args = argparse.Namespace(epics="EURGBP,EURCHF", timeframe="MINUTE_15")
        insts = _instruments_from_args(_config(), args)
        self.assertTrue(all(i.timeframe == "MINUTE_15" for i in insts))

    def test_falls_back_to_config(self):
        args = argparse.Namespace(epics=None, timeframe=None)
        insts = _instruments_from_args(_config(), args)
        self.assertEqual([i.epic for i in insts], ["EURUSD"])


class TestFrozenSetupCheck(unittest.TestCase):
    """demo/live launch must refuse a setup the registry has frozen."""

    def _registry(self, tmpdir, strategy, setup, status):
        from forex_bot.research import ResearchRegistry
        path = Path(tmpdir) / "registry.json"
        reg = ResearchRegistry(path=path)
        reg.set_status(strategy, setup, status, note="test")
        reg.save()
        return path

    def _cfg(self, strategy="rsi_reversion"):
        cfg = TradingConfig(instruments=[InstrumentConfig("EURUSD", "HOUR_4")])
        cfg.strategy = strategy
        return cfg

    def test_frozen_setup_blocks(self):
        from forex_bot.cli import _frozen_setup_check
        with tempfile.TemporaryDirectory() as d:
            path = self._registry(d, "rsi_reversion", "HOUR_4:EURUSD", "holdout-fail")
            reason = _frozen_setup_check(self._cfg(), path)
            self.assertIsNotNone(reason)
            self.assertIn("holdout-fail", reason)

    def test_launchable_statuses_allow(self):
        from forex_bot.cli import _frozen_setup_check
        for status in ("candidate", "holdout-pass", "forward-test", "live"):
            with tempfile.TemporaryDirectory() as d:
                path = self._registry(d, "rsi_reversion", "HOUR_4:EURUSD", status)
                self.assertIsNone(_frozen_setup_check(self._cfg(), path),
                                  f"{status} should be launchable")

    def test_screen_fail_blocks_launch(self):
        # screen-fail is not frozen (it may be re-screened), but it must never
        # launch live/demo: it failed the walk-forward screen.
        from forex_bot.cli import _frozen_setup_check
        with tempfile.TemporaryDirectory() as d:
            path = self._registry(d, "rsi_reversion", "HOUR_4:EURUSD", "screen-fail")
            reason = _frozen_setup_check(self._cfg(), path)
            self.assertIsNotNone(reason)
            self.assertIn("screen-fail", reason)
            # The block message carries strategy, setup, status and evidence.
            self.assertIn("rsi_reversion", reason)
            self.assertIn("HOUR_4:EURUSD", reason)
            self.assertIn("test", reason)  # the entry note

    def test_other_setup_allows(self):
        # Frozen on MINUTE_15 does not block a HOUR_4 launch.
        from forex_bot.cli import _frozen_setup_check
        with tempfile.TemporaryDirectory() as d:
            path = self._registry(d, "rsi_reversion", "MINUTE_15:EURUSD", "holdout-fail")
            self.assertIsNone(_frozen_setup_check(self._cfg(), path))

    def test_missing_registry_allows(self):
        from forex_bot.cli import _frozen_setup_check
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_frozen_setup_check(self._cfg(), Path(d) / "nope.json"))


if __name__ == "__main__":
    unittest.main()

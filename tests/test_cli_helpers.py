import argparse
import unittest

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


if __name__ == "__main__":
    unittest.main()

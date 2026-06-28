import tempfile
import unittest
from pathlib import Path

from forex_bot.config import InstrumentConfig
from forex_bot.research.registry import (
    FROZEN_STATUSES,
    ResearchRegistry,
    setup_key,
)


class TestSetupKey(unittest.TestCase):
    def test_from_instrument_configs_sorted(self):
        insts = [InstrumentConfig(epic="GBPUSD", timeframe="HOUR_4"),
                 InstrumentConfig(epic="AUDUSD", timeframe="HOUR_4")]
        self.assertEqual(setup_key(insts), "HOUR_4:AUDUSD,GBPUSD")

    def test_from_epic_strings_needs_timeframe(self):
        self.assertEqual(
            setup_key(["EURUSD", "AUDUSD"], timeframe="MINUTE_15"),
            "MINUTE_15:AUDUSD,EURUSD",
        )

    def test_order_independent(self):
        a = setup_key(["EURUSD", "AUDUSD"], timeframe="DAY")
        b = setup_key(["AUDUSD", "EURUSD"], timeframe="DAY")
        self.assertEqual(a, b)

    def test_distinguishes_timeframe(self):
        # The whole point: same instruments on a different timeframe is a different
        # setup, so a holdout-fail on M15 does not freeze H4.
        self.assertNotEqual(
            setup_key(["EURUSD"], timeframe="MINUTE_15"),
            setup_key(["EURUSD"], timeframe="HOUR_4"),
        )


class TestRegistryStatus(unittest.TestCase):
    def test_set_and_get(self):
        reg = ResearchRegistry()
        reg.set_status("ema_crossover", "M15:EURUSD", "candidate", note="ok")
        entry = reg.get("ema_crossover", "M15:EURUSD")
        self.assertEqual(entry.status, "candidate")
        self.assertEqual(entry.note, "ok")
        self.assertFalse(entry.frozen)

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            ResearchRegistry().set_status("s", "setup", "totally-bogus")

    def test_frozen_states(self):
        reg = ResearchRegistry()
        reg.set_status("a", "s", "holdout-fail")
        reg.set_status("b", "s", "retired")
        reg.set_status("c", "s", "candidate")
        self.assertTrue(reg.is_frozen("a", "s"))
        self.assertTrue(reg.is_frozen("b", "s"))
        self.assertFalse(reg.is_frozen("c", "s"))
        self.assertEqual(FROZEN_STATUSES, {"holdout-fail", "retired"})

    def test_unknown_entry_not_frozen(self):
        self.assertFalse(ResearchRegistry().is_frozen("nope", "setup"))

    def test_terminal_status_protected_from_auto_overwrite(self):
        reg = ResearchRegistry()
        reg.set_status("rsi", "s", "holdout-fail", note="clean test failed")
        # An automated screen pass must NOT silently re-open a holdout-fail.
        reg.set_status("rsi", "s", "candidate", allow_overwrite_frozen=False)
        self.assertEqual(reg.get("rsi", "s").status, "holdout-fail")
        # A deliberate override still works.
        reg.set_status("rsi", "s", "forward-test", allow_overwrite_frozen=True)
        self.assertEqual(reg.get("rsi", "s").status, "forward-test")

    def test_set_status_is_per_setup(self):
        reg = ResearchRegistry()
        reg.set_status("rsi", "M15:EURUSD", "holdout-fail")
        # Different setup (e.g. H4) is untouched and free to screen.
        self.assertFalse(reg.is_frozen("rsi", "HOUR_4:EURUSD"))


class TestRegistryPersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "registry.json"
            reg = ResearchRegistry(path=path)
            reg.set_status("ema_crossover", "M15:EURUSD", "candidate")
            reg.set_status("rsi", "M15:EURUSD", "holdout-fail", note="failed")
            reg.save()
            self.assertTrue(path.exists())

            reloaded = ResearchRegistry.load(path)
            self.assertEqual(len(reloaded.entries()), 2)
            self.assertEqual(reloaded.get("rsi", "M15:EURUSD").status, "holdout-fail")
            self.assertTrue(reloaded.is_frozen("rsi", "M15:EURUSD"))

    def test_load_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            reg = ResearchRegistry.load(Path(d) / "does_not_exist.json")
            self.assertEqual(reg.entries(), [])


if __name__ == "__main__":
    unittest.main()

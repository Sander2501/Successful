import unittest

from forex_bot.cli import _format_search


class TestSearchFormatting(unittest.TestCase):
    def test_formats_markets(self):
        data = {"markets": [
            {"epic": "EURGBP", "instrumentName": "EUR/GBP", "instrumentType": "CURRENCIES",
             "marketStatus": "TRADEABLE"},
            {"epic": "AUDNZD", "instrumentName": "AUD/NZD", "instrumentType": "CURRENCIES",
             "marketStatus": "TRADEABLE"},
        ]}
        out = _format_search(data)
        self.assertIn("EURGBP", out)
        self.assertIn("AUDNZD", out)
        self.assertIn("TRADEABLE", out)

    def test_empty(self):
        self.assertEqual(_format_search({"markets": []}), "No markets found.")
        self.assertEqual(_format_search({}), "No markets found.")
        self.assertEqual(_format_search(None), "No markets found.")


if __name__ == "__main__":
    unittest.main()

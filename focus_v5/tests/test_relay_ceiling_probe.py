import unittest

from focus_v5.relay_ceiling_probe import summarize


class RelayCeilingSummaryTests(unittest.TestCase):
    def test_summary(self):
        rows = [
            {"regular_rows": 160, "verify_rows": 64, "search": 10, "accepted": 4},
            {"regular_rows": 128, "verify_rows": 64, "search": 8, "accepted": 3},
        ]
        result = summarize(rows)
        self.assertEqual(result["cycles"], 2)
        self.assertEqual(result["baseline_query_rows"], 416)
        self.assertEqual(result["relay_query_rows"], 192)
        self.assertEqual(result["proposals"], 18)
        self.assertEqual(result["accepted"], 7)


if __name__ == "__main__":
    unittest.main()

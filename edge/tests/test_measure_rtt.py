import unittest
from unittest import mock

from scripts.measure_rtt import measure_rtt, summarize_latencies


class MeasureRttTests(unittest.TestCase):
    def test_summarize_latencies_reports_core_percentiles(self):
        summary = summarize_latencies([10.0, 20.0, 30.0, 40.0])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min_ms"], 10.0)
        self.assertEqual(summary["max_ms"], 40.0)
        self.assertEqual(summary["avg_ms"], 25.0)
        self.assertEqual(summary["p50_ms"], 25.0)
        self.assertEqual(summary["p95_ms"], 38.5)

    def test_measure_rtt_collects_successes_and_failures(self):
        with mock.patch(
            "scripts.measure_rtt._measure_once",
            side_effect=[12.5, RuntimeError("boom"), 18.0],
        ):
            result = measure_rtt(
                url="http://example.com/health",
                count=3,
                timeout=1.0,
                warmup=0,
            )

        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["latencies_ms"], [12.5, 18.0])
        self.assertEqual(len(result["errors"]), 1)

    def test_measure_rtt_disables_env_proxy_by_default(self):
        with mock.patch("scripts.measure_rtt._measure_once", return_value=12.5):
            with mock.patch("requests.Session") as session_cls:
                session = session_cls.return_value.__enter__.return_value
                measure_rtt(
                    url="http://example.com/health",
                    count=1,
                    timeout=1.0,
                    warmup=0,
                )

        self.assertFalse(session.trust_env)


if __name__ == "__main__":
    unittest.main()

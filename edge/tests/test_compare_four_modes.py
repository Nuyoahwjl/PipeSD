import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_four_modes.py"
SPEC = importlib.util.spec_from_file_location("compare_four_modes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareFourModesTest(unittest.TestCase):
    def test_writes_one_unscored_completion_file_per_mode(self):
        payload = {
            "manifest": {"run_id": "run-1"},
            "samples": [
                {
                    "task_id": 3,
                    "dataset_task_id": "HumanEval/3",
                    "generated_text": "return 3",
                    "sample_index": 3,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                MODULE.write_completion_jsonl(
                    Path(directory), "humaneval", method, payload
                )
                for method in MODULE.METHODS
            ]
            self.assertEqual(
                {path.name for path in paths},
                {
                    f"humaneval_{method}_completions.jsonl"
                    for method in MODULE.METHODS
                },
            )
            rows = [
                json.loads(line)
                for line in paths[-1].read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            rows,
            [
                {
                    "task_id": "HumanEval/3",
                    "completion": "return 3",
                    "method": "pipesd",
                    "run_id": "run-1",
                    "sample_index": 3,
                }
            ],
        )

    def test_aggregate_network_sums_per_sample_link_totals(self):
        samples = [
            {
                "environment_measurements": {
                    "software_link": {
                        "totals": {
                            "uplink": {
                                "bytes": 100,
                                "transfers": 2,
                                "queue_wait_seconds": 1,
                                "service_seconds": 2,
                            },
                            "downlink": {
                                "bytes": 10,
                                "transfers": 2,
                                "queue_wait_seconds": 0,
                                "service_seconds": 0.5,
                            },
                        }
                    }
                }
            }
        ]
        totals = MODULE.aggregate_network(samples)
        self.assertEqual(totals["uplink_bytes"], 100)
        self.assertEqual(totals["downlink_bytes"], 10)
        self.assertEqual(totals["uplink_transfers"], 2)
        self.assertEqual(totals["downlink_transfers"], 2)
        self.assertEqual(totals["network_transfers"], 4)
        self.assertEqual(totals["network_service_seconds"], 2.5)

    def test_formatter_distinguishes_not_applicable_from_missing(self):
        pure = {"method": "pure_cloud", "network_shaping_mode": None}
        serial = {"method": "vanilla", "network_shaping_mode": "software"}
        self.assertEqual(
            MODULE.fmt_for_method(pure, None, applies_to="collaborative"), "—"
        )
        self.assertEqual(MODULE.fmt_for_method(serial, None), "missing")

    def test_pure_cloud_warning_explains_timing_boundary(self):
        warnings = MODULE.comparability_warnings(
            [
                {
                    "method": "pure_cloud",
                    "actual_tokens": 1000,
                    "seed": 1,
                    "result_tag": "x",
                    "energy_scope": "cloud_gpu",
                    "energy_joules_per_100_tokens": 2,
                }
            ]
        )
        self.assertTrue(any("warm-model local request" in item for item in warnings))

    def test_normalize_result_derives_average_power_from_energy_and_total_time(self):
        payload = {
            "manifest": {
                "algorithm": "pure_cloud",
                "run_id": "run-power",
                "seed": 1,
                "result_tag": "x",
            },
            "summary": {
                "actual_output_tokens": 1000,
                "total_time_seconds": 4.0,
                "weighted_tpt_ms": 4.0,
                "model_energy_joules": 1600.0,
                "model_energy_joules_per_100_tokens": 160.0,
            },
            "samples": [],
        }

        row = MODULE.normalize_result(Path("pure-cloud.json"), payload)

        self.assertEqual(row["average_power_watts"], 400.0)
        self.assertEqual(row["power_time_seconds"], 4.0)
        self.assertEqual(
            row["power_calculation"], "energy_joules / total_time_seconds"
        )

    def test_pure_edge_ignores_legacy_energy_measurements(self):
        payload = {
            "manifest": {
                "algorithm": "pure_edge",
                "run_id": "legacy-rapl",
                "seed": 1,
                "result_tag": "x",
            },
            "summary": {
                "actual_output_tokens": 1000,
                "total_time_seconds": 4.0,
                "model_energy_joules": 100.0,
                "energy_scope": "edge_cpu_package",
                "energy_source": "intel_rapl_package",
            },
            "samples": [],
        }

        row = MODULE.normalize_result(Path("pure-edge.json"), payload)

        self.assertIsNone(row["energy_joules"])
        self.assertIsNone(row["energy_joules_per_100_tokens"])
        self.assertIsNone(row["average_power_watts"])
        self.assertEqual(row["energy_scope"], "not_measured_no_rapl_permission")
        self.assertEqual(row["energy_source"], "not_measured")

    def test_collaborative_tpt_uses_cloud_accepted_tokens(self):
        payload = {
            "manifest": {
                "algorithm": "pipesd",
                "run_id": "run-accepted-tpt",
                "seed": 1,
                "result_tag": "x",
            },
            "summary": {
                "target_accepted_draft_tokens": 1000,
                "actual_accepted_draft_tokens": 1000,
                "actual_output_tokens": 1200,
                "total_time_seconds": 4.0,
                # Deliberately retain a legacy output-normalized value. The
                # comparison script must recompute rather than trust it.
                "weighted_tpt_ms": 4.0 / 1.2,
                "model_energy_joules": 400.0,
                "num_verifications": 200,
            },
            "samples": [],
        }

        row = MODULE.normalize_result(Path("pipesd.json"), payload)

        self.assertEqual(row["actual_tokens"], 1000)
        self.assertEqual(row["actual_output_tokens"], 1200)
        self.assertEqual(
            row["tpt_normalization_token_type"],
            "cloud_accepted_draft_tokens",
        )
        self.assertAlmostEqual(row["tpt_ms"], 4.0)
        self.assertAlmostEqual(row["throughput_tokens_per_second"], 250.0)
        self.assertAlmostEqual(row["energy_joules_per_100_tokens"], 40.0)
        self.assertAlmostEqual(row["nav_per_100_tokens"], 20.0)

    def test_normalize_result_uses_active_energy_window_when_recorded(self):
        payload = {
            "manifest": {
                "algorithm": "pipesd",
                "run_id": "run-active-power",
                "seed": 1,
                "result_tag": "x",
            },
            "summary": {
                "actual_accepted_draft_tokens": 1000,
                "actual_output_tokens": 1000,
                "total_time_seconds": 10.0,
                "weighted_tpt_ms": 10.0,
                "model_energy_joules": 400.0,
                "model_energy_joules_per_100_tokens": 40.0,
                "energy_measurement_duration_seconds": 2.0,
            },
            "samples": [],
        }

        row = MODULE.normalize_result(Path("pipesd.json"), payload)

        self.assertEqual(row["average_power_watts"], 200.0)
        self.assertEqual(row["power_time_seconds"], 2.0)
        self.assertEqual(
            row["power_calculation"],
            "energy_joules / energy_measurement_duration_seconds",
        )

    def test_markdown_explains_thousand_token_tpt_identity_and_power(self):
        rows = [
            {
                "method": "pure_cloud",
                "display_name": "Pure Cloud (model-only)",
                "actual_tokens": 1000,
                "tpt_ms": 4.0,
                "throughput_tokens_per_second": 250.0,
                "total_time_seconds": 4.0,
                "token_latency_p50_ms": 4.0,
                "token_latency_p95_ms": 4.0,
                "token_latency_p99_ms": 4.0,
                "mean_ttft_ms": 4.0,
                "energy_joules_per_100_tokens": 160.0,
                "average_power_watts": 400.0,
                "energy_scope": "cloud_gpu",
                "nav_per_100_tokens": None,
                "mean_draft_length": None,
                "acceptance_rate": None,
                "rollback_rate": None,
                "mean_actual_batch_size": None,
                "uplink_bytes": None,
                "uplink_mib_per_100_tokens": None,
                "uplink_transfers": None,
                "average_uplink_transfer_kib": None,
                "downlink_bytes": None,
                "network_queue_wait_seconds": None,
                "network_service_seconds": None,
                "cap_hit_rate": 1.0,
                "eos_rate": 0.0,
                "run_id": "run-power",
                "git_commit": "abc",
                "seed": 1,
                "network_shaping_mode": None,
                "network_emulator_version": None,
                "uplink_bandwidth_MBps": None,
                "downlink_bandwidth_MBps": None,
                "actual_output_tokens": 1000,
                "tpt_normalization_token_type": "committed_output_tokens",
            }
        ]
        pure_edge = dict(rows[0])
        pure_edge.update(
            {
                "method": "pure_edge",
                "display_name": "Pure Edge (local-only)",
                "energy_joules_per_100_tokens": None,
                "average_power_watts": None,
                "energy_scope": "not_measured_no_rapl_permission",
            }
        )
        rows.append(pure_edge)

        markdown = MODULE.build_markdown("humaneval", rows, [])

        self.assertIn("exactly 1,000 benchmark-normalization tokens", markdown)
        self.assertIn("Collaborative modes use cloud-accepted draft tokens", markdown)
        self.assertIn("Avg power W", markdown)
        self.assertIn("400.000", markdown)
        self.assertIn("| Pure Edge (local-only) | -- | -- |", markdown)

    def test_default_output_is_beside_dataset_experiments(self):
        self.assertEqual(
            MODULE.resolve_output_dir("gsm8k", "four_mode_s1_paper"),
            Path("exp/exp__wjl__four__modes/gsm8k/comparison/four_mode_s1_paper"),
        )


if __name__ == "__main__":
    unittest.main()

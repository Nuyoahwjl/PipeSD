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
        self.assertTrue(any("excludes client-cloud transfer" in item for item in warnings))

    def test_default_output_is_beside_dataset_experiments(self):
        self.assertEqual(
            MODULE.resolve_output_dir("gsm8k", "four_mode_s1_paper"),
            Path("exp/exp__wjl__four__modes/gsm8k/comparison/four_mode_s1_paper"),
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.pure_baseline import (
    GenerationConfig,
    NullEnergyMeter,
    build_summary,
    generate_one,
    load_samples,
    percentile,
    resolve_runtime_defaults,
    select_energy_meter,
)


class FakeModel:
    def __init__(self, tokens, energy_meter=None):
        self.tokens = iter(tokens)
        self.energy_meter = energy_meter

    def reset(self):
        pass

    def tokenize(self, value, add_bos=True):
        return [1, 2]

    def eval(self, tokens):
        if self.energy_meter is not None:
            self.energy_meter.eval_states.append(self.energy_meter.active)
        time.sleep(0.0001)

    def sample(self, **kwargs):
        return next(self.tokens)

    def token_eos(self):
        return 9

    def detokenize(self, tokens):
        return ("|".join(str(token) for token in tokens)).encode()


class RecordingEnergyMeter:
    source = "test_meter"
    available = True

    def __init__(self):
        self.active = False
        self.eval_states = []
        self.last_measurement_duration_seconds = None

    def measure(self, function):
        started = time.perf_counter()
        self.active = True
        try:
            result = function()
        finally:
            self.active = False
            self.last_measurement_duration_seconds = time.perf_counter() - started
        return result, 12.0


class PureBaselineTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 3], 0.5), 2)
        self.assertIsNone(percentile([], 0.5))

    def test_pure_edge_energy_meter_is_always_disabled(self):
        meter = select_energy_meter(
            "pure_edge", gpu_device=0, sample_interval=0.005, n_gpu_layers=0
        )

        self.assertIsInstance(meter, NullEnergyMeter)

    def test_runtime_defaults_match_corresponding_model_roles(self):
        pure_cloud = SimpleNamespace(
            mode="pure_cloud",
            seed=None,
            threads=None,
            ctx_size=None,
            temp=None,
            top_k=None,
            top_p=None,
        )
        pure_edge = SimpleNamespace(
            mode="pure_edge",
            seed=None,
            threads=None,
            ctx_size=None,
            temp=None,
            top_k=None,
            top_p=None,
        )

        resolve_runtime_defaults(pure_cloud)
        resolve_runtime_defaults(pure_edge)

        self.assertEqual(
            (pure_cloud.seed, pure_cloud.threads, pure_cloud.ctx_size),
            (1234, 1, 1024),
        )
        self.assertEqual(
            (pure_cloud.temp, pure_cloud.top_k, pure_cloud.top_p),
            (0.0, 1, 1.0),
        )
        self.assertEqual(
            (pure_edge.seed, pure_edge.threads, pure_edge.ctx_size),
            (1234, 2, 16384),
        )
        self.assertEqual(
            (pure_edge.temp, pure_edge.top_k, pure_edge.top_p),
            (0.0, 1, 0.95),
        )

    def test_generate_one_stops_at_eos_and_has_no_nav(self):
        result = generate_one(
            FakeModel([3, 9, 4]),
            "prompt",
            8,
            GenerationConfig(8, 1, 0.95, 0.0),
            NullEnergyMeter(),
        )
        self.assertEqual(result["output_length"], 2)
        self.assertTrue(result["ended_with_eos"])
        self.assertFalse(result["generation_cap_hit"])
        self.assertEqual(result["verify_stats"]["num_verifications"], 0)
        self.assertEqual(result["generated_text"], "3|9")
        self.assertEqual(len(result["token_durations"]), 2)
        self.assertAlmostEqual(
            result["total_time"],
            result["request_preprocess_time_seconds"]
            + result["prefill_time_seconds"]
            + result["decode_time_seconds"]
            + result["output_postprocess_time_seconds"],
        )
        self.assertGreater(result["total_time"], result["decode_time_seconds"])
        self.assertGreater(
            result["time_to_first_token_seconds"],
            result["token_durations"][0],
        )

    def test_energy_window_covers_prefill_and_complete_decode(self):
        meter = RecordingEnergyMeter()
        result = generate_one(
            FakeModel([3, 9], energy_meter=meter),
            "prompt",
            8,
            GenerationConfig(8, 1, 0.95, 0.0),
            meter,
        )

        # One prompt eval plus one eval for each of the two decoded tokens.
        self.assertEqual(meter.eval_states, [True, True, True])
        self.assertEqual(result["prompt_prefill_gpu_energy_joules"], 12.0)
        self.assertEqual(result["decode_gpu_energy_joules"], 12.0)
        self.assertEqual(result["model_energy_joules"], 24.0)
        self.assertGreater(result["energy_measurement_duration_seconds"], 0.0)
        self.assertEqual(
            result["energy_included_stages"],
            ["prompt_prefill", "autoregressive_decode"],
        )

    def test_summary_uses_na_for_speculative_metrics_and_missing_energy(self):
        sample = {
            "output_length": 2,
            "total_time": 0.2,
            "token_durations": [0.08, 0.12],
            "time_to_first_token_seconds": 0.08,
            "prefill_time_seconds": 0.01,
            "model_energy_joules": None,
            "energy_measurement_duration_seconds": None,
            "energy_source": "unavailable",
            "generation_cap_hit": False,
            "ended_with_eos": True,
            "sample_index": 1,
        }
        summary = build_summary("pure_edge", 2, [sample])
        self.assertAlmostEqual(summary["weighted_tpt_ms"], 100.0)
        self.assertAlmostEqual(summary["throughput_tokens_per_second"], 10.0)
        self.assertIsNone(summary["acceptance_rate"])
        self.assertIsNone(summary["model_energy_joules"])
        self.assertEqual(summary["energy_source"], "not_measured")
        self.assertEqual(
            summary["energy_scope"], "not_measured_no_rapl_permission"
        )

    def test_pure_edge_summary_discards_any_energy_input(self):
        sample = {
            "output_length": 2,
            "total_time": 0.2,
            "token_durations": [0.08, 0.12],
            "time_to_first_token_seconds": 0.09,
            "prefill_time_seconds": 0.01,
            "model_energy_joules": 30.0,
            "energy_measurement_duration_seconds": 0.2,
            "energy_source": "intel_rapl_package",
            "generation_cap_hit": True,
            "ended_with_eos": False,
            "sample_index": 1,
        }

        summary = build_summary("pure_edge", 2, [sample])

        self.assertIsNone(summary["model_energy_joules"])
        self.assertIsNone(summary["edge_energy_joules"])
        self.assertIsNone(summary["energy_measurement_duration_seconds"])
        self.assertEqual(summary["energy_source"], "not_measured")

    def test_pure_cloud_summary_reports_prefill_plus_decode_energy(self):
        sample = {
            "output_length": 2,
            "total_time": 0.2,
            "token_durations": [0.08, 0.12],
            "time_to_first_token_seconds": 0.08,
            "prefill_time_seconds": 0.05,
            "model_energy_joules": 30.0,
            "prompt_prefill_gpu_energy_joules": 5.0,
            "decode_gpu_energy_joules": 25.0,
            "energy_measurement_duration_seconds": 0.25,
            "prompt_prefill_energy_measurement_duration_seconds": 0.05,
            "decode_energy_measurement_duration_seconds": 0.2,
            "energy_source": "nvml_gpu_board_power",
            "generation_cap_hit": True,
            "ended_with_eos": False,
            "sample_index": 1,
        }

        summary = build_summary("pure_cloud", 2, [sample])

        self.assertEqual(
            summary["energy_scope"],
            "cloud_gpu_prompt_prefill_plus_autoregressive_decode",
        )
        self.assertEqual(summary["energy_measurement_duration_seconds"], 0.25)
        self.assertEqual(summary["gpu_energy_joules"], 30.0)
        self.assertEqual(summary["prompt_prefill_gpu_energy_joules"], 5.0)
        self.assertEqual(summary["decode_gpu_energy_joules"], 25.0)
        self.assertEqual(summary["gpu_energy_joules_per_100_tokens"], 1500.0)
        self.assertEqual(summary["average_model_compute_power_watts"], 120.0)

    def test_load_samples_preserves_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gsm8k.jsonl"
            path.write_text(
                json.dumps({"question": "1+1?", "answer": "#### 2"}) + "\n",
                encoding="utf-8",
            )
            samples = load_samples(path, "gsm8k", 0, 0)
        self.assertEqual(samples[0]["prompt"], "1+1?")
        self.assertEqual(samples[0]["reference_answer"], "#### 2")


if __name__ == "__main__":
    unittest.main()

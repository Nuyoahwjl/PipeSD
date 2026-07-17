import json
import tempfile
import time
import unittest
from pathlib import Path

from src.pure_baseline import (
    GenerationConfig,
    NullEnergyMeter,
    build_summary,
    generate_one,
    load_samples,
    percentile,
)


class FakeModel:
    def __init__(self, tokens):
        self.tokens = iter(tokens)

    def reset(self):
        pass

    def tokenize(self, value, add_bos=True):
        return [1, 2]

    def eval(self, tokens):
        time.sleep(0.0001)

    def sample(self, **kwargs):
        return next(self.tokens)

    def token_eos(self):
        return 9

    def detokenize(self, tokens):
        return ("|".join(str(token) for token in tokens)).encode()


class PureBaselineTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 3], 0.5), 2)
        self.assertIsNone(percentile([], 0.5))

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

    def test_summary_uses_na_for_speculative_metrics_and_missing_energy(self):
        sample = {
            "output_length": 2,
            "total_time": 0.2,
            "token_durations": [0.08, 0.12],
            "time_to_first_token_seconds": 0.08,
            "prefill_time_seconds": 0.01,
            "model_energy_joules": None,
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

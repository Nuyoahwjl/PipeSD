import unittest

import numpy as np

from src.local_speculative import (
    build_local_summary,
    generate_one_local_sd,
)
from src.pure_baseline import NullEnergyMeter


class FakeSpeculativeModel:
    """Small llama.cpp-shaped model that always predicts and emits token 1."""

    def __init__(self):
        self.n_tokens = 0
        self.scores = np.zeros((64, 4), dtype=np.float64)

    def reset(self):
        self.n_tokens = 0
        self.scores.fill(0.0)

    def tokenize(self, value, add_bos=True):
        return [0]

    def eval(self, tokens):
        for token in tokens:
            row = np.full(4, -20.0, dtype=np.float64)
            row[1] = 20.0
            self.scores[self.n_tokens] = row
            self.n_tokens += 1

    def sample(self, **kwargs):
        return 1

    def token_eos(self):
        return 3

    def detokenize(self, tokens):
        return ("|".join(str(token) for token in tokens)).encode()


class DifferentNativeTokenizerModel(FakeSpeculativeModel):
    def tokenize(self, value, add_bos=True):
        # Local SD must not call the target tokenizer. The distributed Vanilla
        # path sends canonical draft token IDs directly to the target as well.
        raise AssertionError("target tokenizer must not be used for prompt setup")


class LocalSpeculativeTest(unittest.TestCase):
    def test_target_uses_canonical_draft_prefix_without_retokenizing(self):
        result = generate_one_local_sd(
            FakeSpeculativeModel(),
            DifferentNativeTokenizerModel(),
            "prompt",
            1,
            verify_num=1,
            max_generated_tokens=4,
            seed=3407,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            energy_meter=NullEnergyMeter(),
        )

        self.assertEqual(result["accepted_draft_tokens"], 1)
        self.assertEqual(result["verify_accept_lengths"], [1])

    def test_fixed_window_stops_at_exact_accepted_budget(self):
        result = generate_one_local_sd(
            FakeSpeculativeModel(),
            FakeSpeculativeModel(),
            "prompt",
            3,
            verify_num=2,
            max_generated_tokens=8,
            seed=3407,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            energy_meter=NullEnergyMeter(),
        )

        self.assertEqual(result["accepted_draft_tokens"], 3)
        self.assertEqual(result["verify_spec_lengths"], [2, 1])
        self.assertEqual(result["verify_accept_lengths"], [2, 1])
        self.assertEqual(result["verify_stats"]["num_verifications"], 2)
        # Each NAV also commits one target continuation token.
        self.assertEqual(result["output_length"], 5)
        self.assertEqual(
            [batch["phase"] for batch in result["batch_trace"]],
            ["local_sync_handoff", "local_sync_handoff"],
        )

    def test_summary_uses_accepted_tokens_for_all_local_metrics(self):
        sample = {
            "output_length": 5,
            "total_time": 0.3,
            "token_durations": [0.06] * 5,
            "time_to_first_token_seconds": 0.1,
            "verify_stats": {"num_verifications": 2},
            "verify_spec_lengths": [2, 1],
            "verify_accept_lengths": [2, 1],
            "diagnostics": {"rollback_events": 0},
            "batch_trace": [
                {"actual_batch_size": 2},
                {"actual_batch_size": 1},
            ],
            "model_energy_joules": None,
            "energy_measurement_duration_seconds": None,
            "generation_cap_hit": False,
            "ended_with_eos": False,
            "sample_index": 0,
        }

        summary = build_local_summary("pure_edge", 3, [sample])

        self.assertEqual(summary["actual_accepted_draft_tokens"], 3)
        self.assertEqual(summary["actual_output_tokens"], 5)
        self.assertAlmostEqual(summary["weighted_tpt_ms"], 100.0)
        self.assertAlmostEqual(summary["acceptance_rate"], 1.0)
        self.assertAlmostEqual(summary["mean_draft_length"], 1.5)
        self.assertEqual(
            summary["tpt_normalization_token_type"],
            "target_accepted_draft_tokens",
        )


if __name__ == "__main__":
    unittest.main()

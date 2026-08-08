import threading
import unittest

import numpy as np

from cloud.src.batch_backend import LlamaCppBatchBackend, VerifyRequest, _Session


def logits_for(token, vocab_size=4):
    logits = np.zeros(vocab_size, dtype=np.float64)
    logits[token] = 10.0
    return logits


class LlamaCppBatchBackendStateTests(unittest.TestCase):
    def make_backend(self):
        backend = LlamaCppBatchBackend.__new__(LlamaCppBatchBackend)
        backend._lock = threading.RLock()
        backend._sessions = {}
        backend._free_seq_ids = [0, 1]
        backend.max_sequences = 2
        backend.context_tokens_per_sequence = 32
        backend.decode_batch_tokens = 8
        backend.vocab_size = 4
        backend.removals = []
        backend._memory_seq_rm = lambda seq_id, p0, p1=-1: (
            backend.removals.append((seq_id, p0, p1)) or True
        )
        return backend

    def test_full_accept_keeps_kv_and_defers_final_token(self):
        backend = self.make_backend()
        backend._sessions[11] = _Session(
            task_id=11,
            seq_id=0,
            kv_tokens=[9],
            next_logits=logits_for(1),
        )
        backend._decode_rows = lambda rows: [logits_for(2) for _ in rows]
        draft_probability = np.array([0.0, 0.1, 0.0, 0.0])

        result = backend.verify_batch([
            VerifyRequest(11, 1, [1], [draft_probability], seed=3)
        ])[0]

        self.assertEqual(result['n_accepted'], 1)
        self.assertEqual(result['final_token'], 2)
        self.assertEqual(result['n_past'], 3)
        self.assertEqual(backend._sessions[11].kv_tokens, [9, 1])
        self.assertEqual(backend._sessions[11].pending_final_token, 2)
        self.assertEqual(backend.removals, [])

    def test_rejection_rolls_back_unaccepted_kv_suffix(self):
        backend = self.make_backend()
        backend._sessions[12] = _Session(
            task_id=12,
            seq_id=1,
            kv_tokens=[9],
            next_logits=np.zeros(4),
        )
        backend._decode_rows = lambda rows: [np.zeros(4) for _ in rows]
        draft_probability = np.array([0.0, 1.0, 0.0, 0.0])

        result = backend.verify_batch([
            VerifyRequest(12, 1, [1], [draft_probability], seed=1)
        ])[0]

        self.assertEqual(result['n_accepted'], 0)
        self.assertEqual(backend._sessions[12].kv_tokens, [9])
        self.assertIsNotNone(backend._sessions[12].pending_final_token)
        self.assertEqual(backend.removals, [(1, 1, -1)])

    def test_lazy_rejection_is_resolved_with_one_full_distribution(self):
        backend = self.make_backend()
        backend._sessions[13] = _Session(
            task_id=13,
            seq_id=1,
            kv_tokens=[9],
            next_logits=np.zeros(4),
        )
        backend._decode_rows = lambda rows: [np.zeros(4) for _ in rows]

        pending = backend.verify_batch([
            VerifyRequest(
                13,
                1,
                [1],
                [1.0],
                seed=1,
                prob_transport="lazy_distribution",
            )
        ])[0]

        self.assertEqual(pending['status'], 'needs_full_probs')
        self.assertEqual(pending['rejected_index'], 0)
        self.assertIsNone(backend._sessions[13].pending_final_token)
        self.assertIsNotNone(backend._sessions[13].pending_rejection)

        resolved = backend.resolve_rejection(
            13,
            pending['verification_id'],
            np.array([0.0, 1.0, 0.0, 0.0]),
        )

        self.assertEqual(resolved['status'], 'resolved')
        self.assertEqual(resolved['n_accepted'], 0)
        self.assertIsNotNone(resolved['final_token'])
        self.assertIsNone(backend._sessions[13].pending_rejection)
        self.assertEqual(
            backend._sessions[13].pending_final_token,
            resolved['final_token'],
        )

        baseline = self.make_backend()
        baseline._sessions[13] = _Session(
            task_id=13,
            seq_id=1,
            kv_tokens=[9],
            next_logits=np.zeros(4),
        )
        baseline._decode_rows = lambda rows: [np.zeros(4) for _ in rows]
        full = baseline.verify_batch([
            VerifyRequest(
                13,
                1,
                [1],
                [np.array([0.0, 1.0, 0.0, 0.0])],
                seed=1,
            )
        ])[0]
        self.assertEqual(resolved['final_token'], full['final_token'])

    def test_prefill_interleaves_independent_sequence_ids(self):
        backend = self.make_backend()
        backend.decode_batch_tokens = 4
        decoded_batches = []

        def decode(rows):
            decoded_batches.append(list(rows))
            return [logits_for(token % 4) for token, _, _ in rows]

        backend._decode_rows = decode
        results = backend.initialize_sessions([(21, [1, 2, 3]), (22, [4, 5, 6])])

        self.assertEqual(len(results), 2)
        self.assertEqual(
            decoded_batches[0],
            [(1, 0, 0), (4, 0, 1), (2, 1, 0), (5, 1, 1)],
        )
        self.assertEqual(decoded_batches[1], [(3, 2, 0), (6, 2, 1)])
        self.assertEqual(backend._sessions[21].kv_tokens, [1, 2, 3])
        self.assertEqual(backend._sessions[22].kv_tokens, [4, 5, 6])


if __name__ == '__main__':
    unittest.main()

import threading
import unittest

from cloud.src.batch_backend import VerifyRequest
from cloud.src.batch_scheduler import VerificationBatchScheduler


class FakeBatchBackend:
    def __init__(self):
        self.calls = []
        self.init_calls = []
        self.closed = False

    def estimate_input_tokens(self, request):
        return len(request.tokens)

    def verify_batch(self, requests):
        self.calls.append([request.task_id for request in requests])
        return [
            {
                'task_id': request.task_id,
                'n_accepted': len(request.tokens),
                'n_speculative': len(request.tokens),
                'final_token': 7,
                'n_past': request.n_past + len(request.tokens) + 1,
                'evaluated_tokens': len(request.tokens),
                'seq_id': request.task_id,
            }
            for request in requests
        ]

    def initialize_session(self, task_id, prefix):
        return {'n_past': len(prefix), 'seq_id': task_id}

    def initialize_sessions(self, requests):
        self.init_calls.append([task_id for task_id, _ in requests])
        return [
            {
                'task_id': task_id,
                'n_past': len(prefix),
                'seq_id': task_id,
                'evaluated_tokens': len(prefix),
            }
            for task_id, prefix in requests
        ]

    def close_session(self, task_id):
        return None

    def close(self):
        self.closed = True

    def snapshot(self):
        return {'backend': 'fake'}


class VerificationBatchSchedulerTests(unittest.TestCase):
    def test_concurrent_prefills_share_multi_sequence_decode_work(self):
        backend = FakeBatchBackend()
        scheduler = VerificationBatchScheduler(backend, batch_wait_ms=50)
        barrier = threading.Barrier(3)
        results = []

        def initialize(task_id):
            barrier.wait()
            results.append(scheduler.initialize(task_id, [1, 2, 3]))

        threads = [
            threading.Thread(target=initialize, args=(task_id,))
            for task_id in (1, 2)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(backend.init_calls), 1)
        self.assertTrue(all(result['actual_batch_size'] == 2 for result in results))
        scheduler.shutdown()

    def test_concurrent_clients_share_one_decode_batch(self):
        backend = FakeBatchBackend()
        scheduler = VerificationBatchScheduler(
            backend,
            max_batch_clients=8,
            max_batch_tokens=64,
            batch_wait_ms=50,
        )
        barrier = threading.Barrier(3)
        results = []

        def submit(task_id):
            request = VerifyRequest(
                task_id=task_id,
                n_past=10,
                tokens=[1, 2, 3],
                draft_probs=[],
                seed=1,
            )
            barrier.wait()
            results.append(scheduler.submit(request))

        threads = [threading.Thread(target=submit, args=(task_id,)) for task_id in (1, 2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(set(backend.calls[0]), {1, 2})
        self.assertTrue(all(result['actual_batch_size'] == 2 for result in results))
        self.assertEqual(scheduler.snapshot()['mean_actual_batch_size'], 2.0)
        scheduler.shutdown()
        self.assertTrue(backend.closed)


if __name__ == '__main__':
    unittest.main()

import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CLOUD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CLOUD_ROOT / "src" / "speculative_server.py"


class FakeFastAPI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def post(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def get(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeBaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeEnergyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeLlamaState:
    def __init__(self, history, n_tokens):
        self.history = list(history)
        self.n_tokens = n_tokens
        self.input_ids = np.array(history, dtype=np.intc)
        self.scores = np.zeros((max(1, n_tokens), 8), dtype=np.single)
        self.llama_state = bytes(history)
        self.llama_state_size = len(self.llama_state)
        self.seed = 1234


class FakeLlama:
    def __init__(self, *args, **kwargs):
        self.history = []
        self.n_tokens = 0
        self.scores = np.zeros((0, 8), dtype=np.single)

    def eval(self, tokens):
        self.history.extend(int(token) for token in tokens)
        self.n_tokens = len(self.history)
        if self.n_tokens:
            self.scores = np.zeros((self.n_tokens, 8), dtype=np.single)

    def reset(self):
        self.history = []
        self.n_tokens = 0
        self.scores = np.zeros((0, 8), dtype=np.single)

    def save_state(self):
        return FakeLlamaState(self.history, self.n_tokens)

    def load_state(self, state):
        self.history = list(state.history)
        self.n_tokens = state.n_tokens
        self.scores = np.zeros((max(1, self.n_tokens), 8), dtype=np.single)

    def detokenize(self, tokens):
        return (" ".join(str(token) for token in tokens)).encode("utf-8")

    def sample(self, *args, **kwargs):
        return 1


def install_stub_modules():
    msgpack = types.ModuleType("msgpack")
    msgpack.unpackb = lambda payload, raw=False: payload
    sys.modules["msgpack"] = msgpack

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.HTTPException = FakeHTTPException
    fastapi.Request = object
    sys.modules["fastapi"] = fastapi

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = FakeBaseModel
    sys.modules["pydantic"] = pydantic

    torch = types.ModuleType("torch")
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    torch.cuda = SimpleNamespace(manual_seed=lambda *_: None)
    torch.backends = SimpleNamespace(
        cudnn=SimpleNamespace(deterministic=False, benchmark=False)
    )

    class Generator:
        def manual_seed(self, *_):
            return None

    torch.Generator = Generator
    torch.manual_seed = lambda *_: None
    torch.use_deterministic_algorithms = lambda *_: None
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional

    llama_cpp = types.ModuleType("llama_cpp")
    llama_cpp.Llama = FakeLlama
    llama_cpp.llama_cpp = SimpleNamespace()
    sys.modules["llama_cpp"] = llama_cpp

    util = types.ModuleType("src.util")
    util.seed_everything = lambda *_: None
    util.parse_arguments = lambda: SimpleNamespace(
        seed=1234,
        gamma=4,
        max_tokens=40,
        top_k=1,
        top_p=0.95,
        temp=0.0,
        target_model="fake.gguf",
        threads=1,
        ctx_size=64,
    )
    util.softmax = lambda x, axis=-1: x
    util.max_fn = lambda x: x
    util.sample = lambda probs, *_args, **_kwargs: int(np.argmax(probs))

    class GPUEnergyMonitor:
        def __init__(self, *args, **kwargs):
            self.enabled = False

    util.GPUEnergyMonitor = GPUEnergyMonitor
    util.EnergyTracker = lambda *_args, **_kwargs: FakeEnergyContext()

    src_pkg = types.ModuleType("src")
    src_pkg.util = util
    sys.modules["src"] = src_pkg
    sys.modules["src.util"] = util


def load_module():
    install_stub_modules()
    module_name = "cloud_speculative_server_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.shared_model = module.MyModel("fake.gguf", 64)
    return module


class CloudServerTaskStateTests(unittest.TestCase):
    def test_proc_prefix_measures_only_target_model_eval(self):
        module = load_module()
        args = module.parse_arguments()
        module.gpu_energy_monitor.enabled = True
        stages = []

        class RecordingEnergyContext:
            def __init__(self, _monitor, task, stage, _interval, logger=None):
                self.task = task
                self.stage = stage

            def __enter__(self):
                stages.append(("enter", self.stage))
                return self

            def __exit__(self, exc_type, exc, tb):
                self.task.record_energy_measurement(
                    stage=self.stage,
                    energy_joules=5.0,
                    duration_seconds=0.05,
                    sample_count=3,
                )
                stages.append(("exit", self.stage))
                return False

        module.EnergyTracker = RecordingEnergyContext
        task = module.InferenceTask(task_id=1, prefix=[11, 12, 13], args=args)

        task.proc_prefix()

        self.assertEqual(stages, [("enter", "init_eval"), ("exit", "init_eval")])
        self.assertEqual(task.prompt_prefill_gpu_energy_joules, 5.0)
        self.assertEqual(task.nav_gpu_energy_joules, 0.0)

    def test_energy_accounting_separates_prefill_and_each_nav(self):
        module = load_module()
        args = module.parse_arguments()
        task = module.InferenceTask(task_id=1, prefix=[1, 2], args=args)

        task.record_energy_measurement(
            stage="init_eval", energy_joules=10.0,
            duration_seconds=0.1, sample_count=5,
        )
        task.record_energy_measurement(
            stage="verify_total", energy_joules=3.0,
            duration_seconds=0.02, sample_count=3,
        )
        measurement = task.annotate_last_nav_energy(
            round_id=7,
            n_past=20,
            result={"n_speculative": 4, "n_accepted": 3},
        )

        self.assertEqual(task.total_gpu_power_integral_joules, 13.0)
        self.assertEqual(task.prompt_prefill_gpu_energy_joules, 10.0)
        self.assertEqual(task.nav_gpu_energy_joules, 3.0)
        self.assertAlmostEqual(task.energy_measurement_duration_seconds, 0.12)
        self.assertEqual(len(task.nav_energy_trace), 1)
        self.assertEqual(measurement["speculative_round_id"], 7)
        self.assertEqual(measurement["n_speculative"], 4)
        self.assertEqual(measurement["n_accepted"], 3)

    def test_exit_returns_active_compute_breakdown_and_nav_trace(self):
        module = load_module()
        args = module.parse_arguments()
        module.active_tasks.clear()
        module.gpu_energy_monitor.enabled = True
        task = module.InferenceTask(task_id=1, prefix=[1], args=args)
        task.record_energy_measurement(
            stage="init_eval", energy_joules=5.0,
            duration_seconds=0.05, sample_count=3,
        )
        task.record_energy_measurement(
            stage="verify_total", energy_joules=2.0,
            duration_seconds=0.01, sample_count=2,
        )
        task.annotate_last_nav_energy(
            round_id=0,
            n_past=1,
            result={"n_speculative": 2, "n_accepted": 1},
        )
        module.active_tasks[1] = task

        result = module.handle_exit_payload({"type": "exit", "task_id": 1})

        self.assertEqual(result["model_energy_joules"], 7.0)
        self.assertEqual(result["prompt_prefill_gpu_energy_joules"], 5.0)
        self.assertEqual(result["nav_gpu_energy_joules"], 2.0)
        self.assertEqual(len(result["nav_energy_trace"]), 1)
        self.assertEqual(
            result["energy_scope"],
            "cloud_gpu_prompt_prefill_plus_nav_compute",
        )
        self.assertIn("network_transfer", result["energy_excluded_stages"])

    def test_proc_prefix_snapshots_task_state(self):
        module = load_module()
        args = module.parse_arguments()

        task = module.InferenceTask(task_id=1, prefix=[11, 12, 13], args=args)

        task.proc_prefix()

        self.assertIsNotNone(task.model_state)
        self.assertEqual(task.model_state.n_tokens, 3)

    def test_restore_model_state_recovers_task_specific_context(self):
        module = load_module()
        args = module.parse_arguments()

        task_a = module.InferenceTask(task_id=1, prefix=[1, 2, 3], args=args)
        task_b = module.InferenceTask(task_id=2, prefix=[7, 8], args=args)

        task_a.proc_prefix()
        task_b.proc_prefix()

        module.shared_model.model.eval([99, 100])
        task_a.restore_model_state()

        self.assertEqual(module.shared_model.model.n_tokens, 3)
        self.assertEqual(module.shared_model.model.history, [1, 2, 3])

    def test_inference_task_init_does_not_reset_shared_model(self):
        module = load_module()
        args = module.parse_arguments()

        module.shared_model.model.eval([99])
        self.assertEqual(module.shared_model.model.n_tokens, 1)

        module.InferenceTask(task_id=7, prefix=[1, 2], args=args)

        self.assertEqual(module.shared_model.model.n_tokens, 1)
        self.assertEqual(module.shared_model.model.history, [99])

    def test_handle_propose_payload_allows_parallel_accumulate_for_distinct_tasks(self):
        module = load_module()
        args = module.parse_arguments()
        module.active_tasks.clear()

        task_a = module.InferenceTask(task_id=1, prefix=[1], args=args)
        task_b = module.InferenceTask(task_id=2, prefix=[2], args=args)
        module.active_tasks[1] = task_a
        module.active_tasks[2] = task_b

        def slow_add_batch(tokens, probs, index):
            time.sleep(0.2)
            return True

        task_a.add_batch = slow_add_batch
        task_b.add_batch = slow_add_batch

        payload_a = {"task_id": 1, "tokens": [10], "probs": [[1.0] * 8], "index": 0, "should_verify": False}
        payload_b = {"task_id": 2, "tokens": [20], "probs": [[1.0] * 8], "index": 0, "should_verify": False}

        results = []

        def run(payload):
            results.append(module.handle_propose_payload(payload))

        started = time.perf_counter()
        thread_a = threading.Thread(target=run, args=(payload_a,))
        thread_b = threading.Thread(target=run, args=(payload_b,))
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()
        elapsed = time.perf_counter() - started

        self.assertEqual(len(results), 2)
        self.assertLess(elapsed, 0.35)

    def test_handle_propose_payload_serializes_verify_across_tasks(self):
        module = load_module()
        args = module.parse_arguments()
        module.active_tasks.clear()

        task_a = module.InferenceTask(task_id=1, prefix=[1], args=args)
        task_b = module.InferenceTask(task_id=2, prefix=[2], args=args)
        module.active_tasks[1] = task_a
        module.active_tasks[2] = task_b

        def fast_add_batch(tokens, probs, index):
            return True

        def slow_verify(n_past_at_verify):
            time.sleep(0.2)
            return {
                "n_accepted": 1,
                "n_speculative": 1,
                "final_token": 1,
                "n_past": n_past_at_verify + 1,
            }

        for task in (task_a, task_b):
            task.add_batch = fast_add_batch
            task.restore_model_state = lambda: None
            task.save_model_state = lambda: None
            task.verify_tokens = slow_verify

        payload_a = {"task_id": 1, "tokens": [10], "probs": [[1.0] * 8], "index": 0, "should_verify": True, "n_past": 1}
        payload_b = {"task_id": 2, "tokens": [20], "probs": [[1.0] * 8], "index": 0, "should_verify": True, "n_past": 1}

        results = []

        def run(payload):
            results.append(module.handle_propose_payload(payload))

        started = time.perf_counter()
        thread_a = threading.Thread(target=run, args=(payload_a,))
        thread_b = threading.Thread(target=run, args=(payload_b,))
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()
        elapsed = time.perf_counter() - started

        self.assertEqual(len(results), 2)
        self.assertGreater(elapsed, 0.35)

    def test_stale_proactive_batch_is_discarded_before_buffering(self):
        module = load_module()
        args = module.parse_arguments()
        module.active_tasks.clear()
        task = module.InferenceTask(task_id=1, prefix=[1], args=args)
        task.last_verify_pass = False
        task.final_token = 4
        module.active_tasks[1] = task

        result = module.handle_propose_payload({
            "type": "propose_waiting",
            "task_id": 1,
            "tokens": [5, 6],
            "probs": [[1.0] * 8, [1.0] * 8],
            "index": 0,
            "should_verify": False,
            "expected_prefix_token": 4,
        })

        self.assertEqual(result["status"], "discarded_stale_proactive_batch")
        self.assertEqual(task.accumulated_tokens, [])
        self.assertEqual(task.discarded_proactive_tokens, 2)

    def test_matching_proactive_batch_is_buffered(self):
        module = load_module()
        args = module.parse_arguments()
        module.active_tasks.clear()
        task = module.InferenceTask(task_id=1, prefix=[1], args=args)
        task.last_verify_pass = True
        task.final_token = 4
        module.active_tasks[1] = task

        result = module.handle_propose_payload({
            "type": "propose_waiting",
            "task_id": 1,
            "tokens": [5, 6],
            "probs": [[1.0] * 8, [1.0] * 8],
            "index": 0,
            "should_verify": False,
            "expected_prefix_token": 4,
        })

        self.assertEqual(result["status"], "accumulated")
        self.assertEqual(task.accumulated_tokens, [5, 6])
        self.assertEqual(task.reused_proactive_tokens, 2)

    def test_proactive_batch_arrives_while_parent_nav_is_running(self):
        module = load_module()
        args = module.parse_arguments()
        module.active_tasks.clear()
        task = module.InferenceTask(task_id=1, prefix=[1], args=args)
        module.active_tasks[1] = task
        verify_started = threading.Event()
        allow_verify_to_finish = threading.Event()

        task.restore_model_state = lambda: None
        task.save_model_state = lambda: None

        def slow_verify(n_past_at_verify):
            verify_started.set()
            self.assertTrue(allow_verify_to_finish.wait(timeout=2.0))
            return {
                "n_accepted": 1,
                "n_speculative": 1,
                "final_token": 4,
                "n_past": n_past_at_verify + 2,
            }

        task.verify_tokens = slow_verify
        nav_result = {}

        def run_nav():
            nav_result.update(module.handle_propose_payload({
                "type": "propose",
                "task_id": 1,
                "tokens": [3],
                "probs": [[1.0] * 8],
                "index": 0,
                "should_verify": True,
                "n_past": 1,
                "speculative_round_id": 0,
            }))

        nav_thread = threading.Thread(target=run_nav)
        nav_thread.start()
        self.assertTrue(verify_started.wait(timeout=1.0))

        started = time.perf_counter()
        proactive_result = module.handle_propose_payload({
            "type": "propose_waiting",
            "task_id": 1,
            "tokens": [5],
            "probs": [[1.0] * 8],
            "index": 0,
            "should_verify": False,
            "n_past": 3,
            "speculative_round_id": 1,
            "parent_round_id": 0,
            "prefix_version": 1,
            "expected_prefix_token": 4,
        })
        elapsed = time.perf_counter() - started

        self.assertEqual(proactive_result["status"], "buffered_pending_nav")
        self.assertLess(elapsed, 0.1)
        self.assertTrue(nav_thread.is_alive())

        allow_verify_to_finish.set()
        nav_thread.join(timeout=2.0)
        self.assertFalse(nav_thread.is_alive())
        self.assertEqual(nav_result["final_token"], 4)
        self.assertEqual(task.accumulated_tokens, [5])
        self.assertEqual(task.reused_proactive_tokens, 1)

    def test_waiting_nav_may_arrive_before_parent_nav_starts(self):
        module = load_module()
        args = module.parse_arguments()
        module.active_tasks.clear()
        task = module.InferenceTask(task_id=1, prefix=[1], args=args)
        module.active_tasks[1] = task
        task.restore_model_state = lambda: None
        task.save_model_state = lambda: None
        verify_calls = []

        def verify(n_past_at_verify):
            speculative_count = len([
                token for token in task.accumulated_tokens if token is not None
            ])
            verify_calls.append(speculative_count)
            task.reset_accumulated()
            if len(verify_calls) == 1:
                return {
                    "n_accepted": 1,
                    "n_speculative": 1,
                    "final_token": 4,
                    "n_past": n_past_at_verify + 2,
                }
            return {
                "n_accepted": 1,
                "n_speculative": 1,
                "final_token": 6,
                "n_past": n_past_at_verify + 2,
            }

        task.verify_tokens = verify
        waiting_result = {}

        def run_waiting_nav():
            waiting_result.update(module.handle_propose_payload({
                "type": "propose_waiting",
                "task_id": 1,
                "tokens": [5],
                "probs": [[1.0] * 8],
                "index": 0,
                "should_verify": True,
                "n_past": 3,
                "speculative_round_id": 1,
                "parent_round_id": 0,
                "prefix_version": 1,
                "expected_prefix_token": 4,
            }))

        waiting_thread = threading.Thread(target=run_waiting_nav)
        waiting_thread.start()

        deadline = time.time() + 1.0
        while 0 not in task.proactive_buffers and time.time() < deadline:
            time.sleep(0.005)
        self.assertIn(0, task.proactive_buffers)
        self.assertTrue(waiting_thread.is_alive())

        parent_result = module.handle_propose_payload({
            "type": "propose",
            "task_id": 1,
            "tokens": [3],
            "probs": [[1.0] * 8],
            "index": 0,
            "should_verify": True,
            "n_past": 1,
            "speculative_round_id": 0,
        })

        waiting_thread.join(timeout=2.0)
        self.assertFalse(waiting_thread.is_alive())
        self.assertEqual(parent_result["n_accepted"], 1)
        self.assertEqual(waiting_result["n_accepted"], 1)
        self.assertNotEqual(
            waiting_result.get("status"), "discarded_stale_proactive_batch"
        )
        self.assertEqual(verify_calls, [1, 1])
        self.assertEqual(task.reused_proactive_tokens, 1)


if __name__ == "__main__":
    unittest.main()

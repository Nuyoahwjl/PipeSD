import json
import importlib
import os
import sys
import tempfile
import types
import unittest
import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def install_stub_modules():
    torch = types.ModuleType("torch")

    def no_grad():
        def decorator(fn):
            return fn
        return decorator

    torch.no_grad = no_grad
    torch.multiprocessing = types.ModuleType("torch.multiprocessing")
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = torch
    sys.modules["torch.multiprocessing"] = torch.multiprocessing
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional

    transformers = types.ModuleType("transformers")
    transformers.utils = SimpleNamespace(logging=SimpleNamespace(set_verbosity=lambda *_: None))
    sys.modules["transformers"] = transformers

    skopt = types.ModuleType("skopt")
    skopt.gp_minimize = lambda *args, **kwargs: None
    skopt_space = types.ModuleType("skopt.space")
    skopt_space.Real = lambda *args, **kwargs: ("Real", args, kwargs)
    sys.modules["skopt"] = skopt
    sys.modules["skopt.space"] = skopt_space

    msgpack = types.ModuleType("msgpack")
    msgpack.packb = lambda payload: payload
    sys.modules["msgpack"] = msgpack

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = object
    sys.modules["pandas"] = pandas

    llama_cpp = types.ModuleType("llama_cpp")

    class StubLlama:
        def __init__(self, *args, **kwargs):
            self.n_tokens = 0

    llama_cpp.Llama = StubLlama
    sys.modules["llama_cpp"] = llama_cpp


install_stub_modules()

from app.run_edge import CloudEdgeSpeculativeEval
from src.engine import Decoding


class FakeSender:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class BootstrapSender(FakeSender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submissions = []

    def submit(self, url, payload, headers=None, tag=None, token_count=None):
        self.submissions.append((url, payload, token_count))
        callback = self.kwargs.get("on_complete")
        if callback is not None:
            callback({
                "success": True,
                "token_count": token_count,
                "elapsed_seconds": 0.05 + 0.01 * token_count,
            })
        future = concurrent.futures.Future()
        future.set_result({"body_size_bytes": len(payload)})
        return future


class FakeLlama:
    def __init__(self, *args, **kwargs):
        self.n_tokens = 0
        self.kwargs = kwargs


class DummyDecoding(Decoding):
    def load_data(self):
        return []

    def preprocess(self, input_text):
        return input_text, 0

    def postprocess(self, input_text, output_text):
        return output_text


def make_args(**overrides):
    base = dict(
        seed=1,
        gamma=6,
        max_generated_tokens=8,
        top_k=1,
        top_p=0.95,
        temp=0.0,
        C=0.05,
        verify_strategy="fixed-num",
        verify_num=3,
        bandwidth_MBps=2.5,
        multiply_times=0.95,
        algorithm="vanilla",
        start_index_of_sample=0,
        end_index_of_sample=0,
        dataset="gsm8k",
        verify_thresh_single=0.94,
        verify_thresh_multi=0.9,
        init_alpha=0.92,
        draft_model="fake.gguf",
        threads=1,
        ctx_size=64,
        use_env_proxy=False,
        server_timeout_s=10,
        ablation_study=False,
        bayes_optimize=False,
        bayes_calls=15,
        bayes_single_min=0.6,
        bayes_single_max=0.99,
        bayes_multi_min=0.05,
        bayes_multi_max=0.9,
        nomerge=False,
        default_token_compute=0.036,
        token_size_MB=0.29,
        schedule_window=20,
        schedule_history_size=100,
        environment_update_threshold=0.2,
        regression_min_comm_samples=8,
        disable_online_environment_measurement=False,
        merge_policy="dp",
        result_tag="",
        task_id_offset=0,
        draft_n_gpu_layers=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class RunEdgeTests(unittest.TestCase):
    def test_bayes_latency_trial_does_not_reuse_previous_candidate_tokens(self):
        evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
        evaluator.samples = ["sample"]
        evaluator.args = SimpleNamespace(verify_thresh_single=0.0, verify_thresh_multi=0.0)
        evaluator._token_durations = [99.0]
        evaluator.preprocess = lambda sample: (sample, 0)
        evaluator._reset_state = lambda: None

        def run_sample(*args, **kwargs):
            evaluator._token_durations.extend([0.1] * 25)

        evaluator.edge_process_draft_model = run_sample

        latencies = evaluator._run_latency_trial(0.4, 0.6, min_tokens=20)

        self.assertEqual(latencies, [0.1] * 25)
        self.assertNotIn(99.0, latencies)

    def test_load_data_respects_max_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "gsm8k.jsonl"
            rows = [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
            ]
            data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
            evaluator.args = SimpleNamespace(dataset="gsm8k", data_path=str(data_path), max_samples=1)
            evaluator.start_index_of_sample = 0
            evaluator.end_index_of_sample = 4
            evaluator.color_print = lambda *args, **kwargs: None

            samples = CloudEdgeSpeculativeEval.load_data(evaluator)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["prompt"], "q1")

    def test_preprocess_applies_task_id_offset(self):
        evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
        evaluator.args = SimpleNamespace(dataset="gsm8k", task_id_offset=1000)

        prompt, task_id = CloudEdgeSpeculativeEval.preprocess(
            evaluator,
            {"prompt": "q1", "task_id": 7},
        )

        self.assertEqual(prompt, "q1")
        self.assertEqual(task_id, 1007)

    def test_reset_state_initializes_tracking_fields(self):
        args = make_args()

        decoder = DummyDecoding(args)
        decoder.color_print = lambda *args, **kwargs: None

        with mock.patch("src.engine.Llama", FakeLlama), mock.patch("src.engine.BandwidthSender", FakeSender):
            decoder._reset_state()

        self.assertEqual(decoder.verify_num, 3)
        self.assertEqual(decoder._spec_token_indices_generated, [])
        self.assertEqual(decoder._spec_token_indices_sent, set())
        self.assertFalse(decoder.sender.kwargs["use_env_proxy"])

    def test_reset_state_forwards_use_env_proxy_to_sender(self):
        args = make_args(use_env_proxy=True)

        decoder = DummyDecoding(args)
        decoder.color_print = lambda *args, **kwargs: None

        with mock.patch("src.engine.Llama", FakeLlama), mock.patch("src.engine.BandwidthSender", FakeSender):
            decoder._reset_state()

        self.assertTrue(decoder.sender.kwargs["use_env_proxy"])

    def test_reset_state_forwards_server_timeout_to_sender(self):
        args = make_args(server_timeout_s=45)

        decoder = DummyDecoding(args)
        decoder.color_print = lambda *args, **kwargs: None

        with mock.patch("src.engine.Llama", FakeLlama), mock.patch("src.engine.BandwidthSender", FakeSender):
            decoder._reset_state()

        self.assertEqual(decoder.sender.kwargs["timeout"], 45)

    def test_reset_state_forwards_draft_n_gpu_layers_to_llama(self):
        args = make_args(draft_n_gpu_layers=-1)

        decoder = DummyDecoding(args)
        decoder.color_print = lambda *args, **kwargs: None

        with mock.patch("src.engine.Llama", FakeLlama), mock.patch("src.engine.BandwidthSender", FakeSender):
            decoder._reset_state()

        self.assertEqual(decoder.draft_model.kwargs["n_gpu_layers"], -1)

    def test_record_token_time_appends_per_token_durations(self):
        args = make_args()
        decoder = DummyDecoding(args)
        decoder._token_durations = []
        decoder._token_time_ref = 100.0

        with mock.patch("src.engine.time.time", return_value=106.0):
            decoder._record_token_time(3)

        self.assertEqual(decoder._token_durations, [2.0, 2.0, 2.0])
        self.assertEqual(decoder._token_time_ref, 106.0)

    def test_pipesd_bootstraps_communication_with_batch_sizes_one_to_eight(self):
        args = make_args(
            algorithm="pipesd",
            verify_strategy="hybrid",
            token_size_MB=0.000001,
        )
        decoder = DummyDecoding(args)
        decoder.color_print = lambda *args, **kwargs: None

        with mock.patch("src.engine.Llama", FakeLlama), mock.patch("src.engine.BandwidthSender", BootstrapSender):
            decoder._reset_state()

        self.assertEqual([item[2] for item in decoder.sender.submissions], list(range(1, 9)))
        self.assertEqual([len(item[1]) for item in decoder.sender.submissions], list(range(1, 9)))
        estimates = decoder.environment_estimator.estimate()
        self.assertAlmostEqual(estimates["alpha"], 0.05, places=6)
        self.assertAlmostEqual(estimates["beta"], 0.01, places=6)

    def test_resolve_merge_plan_supports_immediate_policy(self):
        decoder = DummyDecoding(make_args(algorithm="pipesd", verify_strategy="hybrid", merge_policy="immediate"))

        merge_plan = decoder._resolve_merge_plan()

        self.assertEqual(merge_plan, [1] * 40)

    def test_resolve_merge_plan_uses_paper_scheduling_window(self):
        decoder = DummyDecoding(make_args(algorithm="pipesd", verify_strategy="hybrid"))

        merge_plan = decoder._resolve_merge_plan()

        self.assertEqual(sum(merge_plan), 20)

    def test_exp2path_distinguishes_pipesd_merge_policy(self):
        decoder = DummyDecoding(
            make_args(
                algorithm="pipesd",
                verify_strategy="hybrid",
                merge_policy="no_early",
                result_tag="nav_diag_pilot",
            )
        )

        saved_path = decoder.exp2path("2.5")

        self.assertIn("merge=no_early", saved_path)
        self.assertIn("tag=nav_diag_pilot", saved_path)

    def test_build_verify_diagnostics_reports_rollback_and_frequency(self):
        decoder = DummyDecoding(make_args())
        decoder.verify_spec_lengths = [4, 2, 5]
        decoder.verify_accept_lengths = [4, 1, 3]

        diagnostics = decoder._build_verify_diagnostics(output_length=10)

        self.assertEqual(diagnostics["mean_verify_spec_len"], 11 / 3)
        self.assertEqual(diagnostics["mean_accept_len"], 8 / 3)
        self.assertEqual(diagnostics["mean_rejected_len"], 1.0)
        self.assertEqual(diagnostics["rollback_events"], 2)
        self.assertEqual(diagnostics["rollback_rate"], 2 / 3)
        self.assertEqual(diagnostics["verification_frequency"], 0.3)
        self.assertEqual(diagnostics["draft_length_hist"], {"4": 1, "2": 1, "5": 1})
        self.assertEqual(diagnostics["accepted_length_hist"], {"4": 1, "1": 1, "3": 1})
        self.assertEqual(diagnostics["rejected_length_hist"], {"0": 1, "1": 1, "2": 1})

    def test_resolve_waiting_verify_length_uses_full_waiting_sequence_when_no_rebatch(self):
        decoder = DummyDecoding(make_args())

        waiting_spec_len = decoder._resolve_waiting_verify_length(
            waiting_tokens=[1, 2, 3, 4],
            waiting_batch_tokens=None,
        )

        self.assertEqual(waiting_spec_len, 4)

    def test_resolve_waiting_verify_length_uses_full_waiting_sequence_even_with_rebatch(self):
        decoder = DummyDecoding(make_args())

        waiting_spec_len = decoder._resolve_waiting_verify_length(
            waiting_tokens=[1, 2, 3, 4, 5],
            waiting_batch_tokens=[4, 5],
        )

        self.assertEqual(waiting_spec_len, 5)

    def test_engine_url_respects_env_override(self):
        import src.engine as engine_module

        previous = os.environ.get("PIPE_SD_SERVER_URL")
        os.environ["PIPE_SD_SERVER_URL"] = "http://127.0.0.1:1597"
        try:
            engine_module = importlib.reload(engine_module)
            self.assertEqual(engine_module.URL, "http://127.0.0.1:1597")
            self.assertEqual(engine_module.INIT_ENDPOINT, "http://127.0.0.1:1597/init")
            self.assertEqual(engine_module.PROPOSE_ENDPOINT, "http://127.0.0.1:1597/propose")
            self.assertEqual(engine_module.EXIT_ENDPOINT, "http://127.0.0.1:1597/exit")
        finally:
            if previous is None:
                os.environ.pop("PIPE_SD_SERVER_URL", None)
            else:
                os.environ["PIPE_SD_SERVER_URL"] = previous
            importlib.reload(engine_module)


if __name__ == "__main__":
    unittest.main()

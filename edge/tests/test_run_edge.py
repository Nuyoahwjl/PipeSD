import json
import importlib
import os
import sys
import tempfile
import types
import unittest
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
        merge_policy="dp",
        result_tag="",
        task_id_offset=0,
        draft_n_gpu_layers=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class RunEdgeTests(unittest.TestCase):
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

    def test_resolve_merge_plan_supports_immediate_policy(self):
        decoder = DummyDecoding(make_args(algorithm="pipesd", verify_strategy="hybrid", merge_policy="immediate"))

        merge_plan = decoder._resolve_merge_plan()

        self.assertEqual(merge_plan, [1] * 40)

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

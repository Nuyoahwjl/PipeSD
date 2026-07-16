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

    def submit(self, url, payload, headers=None, tag=None, token_count=None, measurement_kind=None):
        self.submissions.append((url, payload, token_count))
        callback = self.kwargs.get("on_complete")
        if callback is not None:
            callback({
                "success": True,
                "token_count": token_count,
                "measurement_kind": measurement_kind,
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

        observed_budgets = []

        def run_sample(*args, **kwargs):
            budget = kwargs["max_accepted_tokens"]
            observed_budgets.append(budget)
            evaluator._token_durations.extend([0.1] * budget)

        evaluator.edge_process_draft_model = run_sample

        latencies = evaluator._run_latency_trial(0.4, 0.6, tokens_per_sample=20)

        self.assertEqual(latencies, [0.1] * 20)
        self.assertEqual(observed_budgets, [20])
        self.assertNotIn(99.0, latencies)

    def test_bayes_latency_trial_collects_twenty_tokens_from_each_of_ten_samples(self):
        evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
        evaluator.samples = list(range(10))
        evaluator.args = SimpleNamespace(verify_thresh_single=0.0, verify_thresh_multi=0.0)
        evaluator._token_durations = []
        evaluator.preprocess = lambda sample: (f"prompt-{sample}", sample)
        evaluator._reset_state = lambda: None
        observed = []

        def run_sample(prompt, task_id, **kwargs):
            budget = kwargs["max_accepted_tokens"]
            observed.append((prompt, task_id, budget))
            evaluator._token_durations.extend([0.1] * budget)

        evaluator.edge_process_draft_model = run_sample

        latencies = evaluator._run_latency_trial(0.4, 0.6, tokens_per_sample=20)

        self.assertEqual(len(latencies), 200)
        self.assertEqual([item[1] for item in observed], list(range(10)))
        self.assertEqual([item[2] for item in observed], [20] * 10)

    def test_paper_bayes_trial_collects_twenty_tokens_total(self):
        evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
        evaluator.samples = [
            {"prompt": f"prompt-{index}", "task_id": index, "sample_index": index}
            for index in range(10)
        ]
        evaluator.args = SimpleNamespace(
            verify_thresh_single=0.0,
            verify_thresh_multi=0.0,
            bo_protocol="paper",
        )
        evaluator.preprocess = lambda sample: (sample["prompt"], sample["task_id"])
        evaluator._reset_state = lambda: None
        observed_budgets = []

        def run_sample(prompt, task_id, **kwargs):
            budget = kwargs["max_accepted_tokens"]
            observed_budgets.append(budget)
            evaluator._token_durations.extend([0.1] * budget)

        evaluator.edge_process_draft_model = run_sample

        latencies = evaluator._run_latency_trial(0.4, 0.6, token_budget=20)

        self.assertEqual(len(latencies), 20)
        self.assertEqual(observed_budgets, [20])

    def test_edgellm_rejects_bayesian_optimization(self):
        evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
        evaluator.algorithm = "edgeLLM"

        with self.assertRaisesRegex(ValueError, "supported only for PipeSD"):
            evaluator.bayes_optimize_thresholds()

    def test_run_summary_uses_token_weighted_tpt(self):
        evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
        evaluator.args = SimpleNamespace(evaluation_protocol="paper_table1", target_output_tokens=10)
        results = [
            {"output_length": 2, "total_time": 1.0, "token_durations": [0.5, 0.5]},
            {"output_length": 8, "total_time": 2.0, "token_durations": [0.25] * 8},
        ]

        summary = evaluator._build_run_summary(results)

        self.assertEqual(summary["actual_output_tokens"], 10)
        self.assertAlmostEqual(summary["weighted_tpt_seconds"], 0.3)

    def test_paper_table1_stops_exactly_at_token_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = CloudEdgeSpeculativeEval.__new__(CloudEdgeSpeculativeEval)
            evaluator.samples = [
                {"prompt": "p0", "task_id": 0, "sample_index": 0, "reference_answer": "#### 1"},
                {"prompt": "p1", "task_id": 1, "sample_index": 1, "reference_answer": "#### 2"},
            ]
            evaluator.args = SimpleNamespace(
                target_output_tokens=10,
                dataset="gsm8k",
                evaluation_protocol="paper_table1",
            )
            evaluator.preprocess = lambda sample: (sample["prompt"], sample["task_id"])
            evaluator._reset_state = lambda: None
            evaluator._build_manifest = lambda: {"run_id": "test"}
            evaluator._paper_result_path = lambda: Path(tmpdir) / "result.json"
            evaluator.color_print = lambda *args, **kwargs: None
            budgets = []

            def run_sample(prompt, task_id, **kwargs):
                budget = kwargs["max_accepted_tokens"]
                budgets.append(budget)
                produced = min(6, budget)
                return {
                    "task_id": task_id,
                    "output_length": produced,
                    "total_time": float(produced),
                    "token_durations": [1.0] * produced,
                }

            evaluator.edge_process_draft_model = run_sample

            payload = evaluator._run_paper_table1()

            self.assertEqual(budgets, [10, 4])
            self.assertEqual(payload["summary"]["actual_output_tokens"], 10)
            self.assertEqual([sample["output_length"] for sample in payload["samples"]], [6, 4])
            self.assertTrue((Path(tmpdir) / "result.json").exists())

    def test_compute_emulation_is_disabled_by_default(self):
        decoder = DummyDecoding(make_args())

        with mock.patch("src.engine.time.sleep") as sleep:
            decoder._apply_compute_emulation()

        sleep.assert_not_called()

    def test_compute_emulation_uses_only_explicit_extra_delay(self):
        decoder = DummyDecoding(make_args(enable_compute_emulation=True, emulated_generation_delay=0.012))

        with mock.patch("src.engine.time.sleep") as sleep:
            decoder._apply_compute_emulation()

        sleep.assert_called_once_with(0.012)

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
        self.assertIsNot(decoder.sender, decoder.proactive_sender)
        self.assertIs(decoder.sender.kwargs["link"], decoder.proactive_sender.kwargs["link"])
        self.assertIs(decoder.sender.kwargs["link"], decoder.software_link)
        self.assertEqual(decoder.software_link.snapshot()["uplink_bandwidth_MBps"], 2.5)
        self.assertEqual(decoder.software_link.snapshot()["downlink_bandwidth_MBps"], 25.0)

    def test_software_network_configuration_records_startup_and_profile(self):
        decoder = DummyDecoding(make_args(
            C=0.025,
            software_uplink_startup_ms=30.0,
            software_downlink_startup_ms=5.0,
            software_bandwidth_profile="1.25:18.75,10:35",
            software_bandwidth_change_interval_s=20.0,
        ))

        snapshot = decoder._network_configuration_snapshot()

        self.assertEqual(snapshot["mode"], "software")
        self.assertEqual(snapshot["uplink_startup_seconds"], 0.03)
        self.assertEqual(snapshot["downlink_startup_seconds"], 0.005)
        self.assertEqual(snapshot["bandwidth_profile"], [[1.25, 18.75], [10.0, 35.0]])
        self.assertEqual(snapshot["bandwidth_change_interval_seconds"], 20.0)

    def test_latest_bayes_config_records_software_link_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decoder = DummyDecoding(make_args(run_id="bo-test"))
            decoder.exp_name = tmpdir

            path = CloudEdgeSpeculativeEval._write_latest_bayes_config(decoder, {
                "best_thresh_single": 0.8,
                "best_thresh_multi": 0.5,
            })
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["network_emulation"]["emulator_version"], "shared-fifo-v1")
        self.assertEqual(payload["network_emulation"]["queue_policy"], "shared-fifo-per-direction")

    def test_nav_measurement_is_excluded_from_communication_regression(self):
        decoder = DummyDecoding(make_args(algorithm="pipesd", verify_strategy="hybrid"))
        before = len(decoder.environment_estimator.comm_samples)

        decoder._on_send_measurement({
            "success": True,
            "measurement_kind": "nav",
            "token_count": 4,
            "elapsed_seconds": 10.0,
        })

        self.assertEqual(len(decoder.environment_estimator.comm_samples), before)

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

    def test_commit_verified_tokens_respects_remaining_generation_budget(self):
        decoder = DummyDecoding(make_args())
        decoder.max_len = 6
        decoder._token_durations = []
        decoder._token_time_ref = 100.0
        output_tokens = [10, 11, 12, 13]

        with mock.patch("src.engine.time.time", return_value=102.0):
            committed = decoder._commit_verified_tokens(
                output_tokens,
                speculative_tokens=[20, 21, 22],
                n_accepted=3,
                final_token=30,
            )

        self.assertEqual(committed, 2)
        self.assertEqual(output_tokens, [10, 11, 12, 13, 20, 21])
        self.assertEqual(len(decoder._token_durations), 2)

    def test_generation_budget_forces_verification_before_final_token_slot(self):
        decoder = DummyDecoding(make_args())
        decoder.max_len = 6

        self.assertFalse(decoder._must_verify_for_budget([1, 2, 3, 4], []))
        self.assertTrue(decoder._must_verify_for_budget([1, 2, 3, 4], [5]))
        self.assertTrue(decoder._must_verify_for_budget([1, 2, 3], [4], [5]))

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
        self.assertEqual(
            [len(item[1]["probs"]) for item in decoder.sender.submissions],
            list(range(1, 9)),
        )
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

    def test_only_pipesd_uses_pre_nav_batch_uploads(self):
        self.assertFalse(DummyDecoding(make_args(algorithm="vanilla"))._uses_pre_nav_pipeline())
        self.assertFalse(DummyDecoding(make_args(algorithm="hsl"))._uses_pre_nav_pipeline())
        self.assertFalse(DummyDecoding(make_args(algorithm="edgeLLM"))._uses_pre_nav_pipeline())
        self.assertTrue(DummyDecoding(make_args(algorithm="pipesd"))._uses_pre_nav_pipeline())
        self.assertFalse(
            DummyDecoding(make_args(algorithm="pipesd", nomerge=True))._uses_pre_nav_pipeline()
        )

    def test_edgellm_waiting_batches_use_current_nhat_without_dp(self):
        decoder = DummyDecoding(make_args(algorithm="edgeLLM"))
        decoder.dp_scheduler.window = 7

        with mock.patch.object(decoder, "_resolve_merge_plan") as resolve_dp:
            plan = decoder._resolve_algorithm_batch_plan()

        self.assertEqual(plan, [7])
        resolve_dp.assert_not_called()

    def test_edgellm_threshold_update_compares_accepts_with_nhat(self):
        decoder = DummyDecoding(
            make_args(algorithm="edgeLLM", edge_llm_full_accept_decay=0.5)
        )
        decoder.dp_scheduler.window = 5
        decoder.alpha = 0.4
        decoder.accumulated_probs = 0.25

        # Full acceptance of a short, two-token draft is still N_correct < N-hat.
        decoder.update_thresh(multiply_times=0.01, n_accepted=2, n_all=2)
        self.assertAlmostEqual(decoder.alpha, 0.4 / (0.25 ** 0.6))

        decoder.alpha = 0.4
        decoder.update_thresh(multiply_times=0.01, n_accepted=5, n_all=8)
        self.assertAlmostEqual(decoder.alpha, 0.2)

    def test_edgellm_threshold_update_uses_confidence_snapshot_for_its_nav(self):
        decoder = DummyDecoding(
            make_args(algorithm="edgeLLM", edge_llm_full_accept_decay=0.5)
        )
        decoder.dp_scheduler.window = 5
        decoder.alpha = 0.2
        decoder.accumulated_probs = 0.01  # confidence overwritten by proactive generation

        decoder.update_thresh(
            multiply_times=0.01,
            n_accepted=2,
            n_all=3,
            accumulated_probs=0.25,
        )

        self.assertAlmostEqual(decoder.alpha, 0.2 / (0.25 ** 0.6))

    def test_edgellm_result_name_records_paper_decay_not_legacy_multiplier(self):
        decoder = DummyDecoding(
            make_args(
                algorithm="edgeLLM",
                init_alpha=0.92,
                multiply_times=0.17,
                edge_llm_full_accept_decay=0.5,
            )
        )

        saved_path = decoder.exp2path("2.5")

        self.assertIn("alpha=0.92_decay=0.5", saved_path)
        self.assertNotIn("mult=", saved_path)

    def test_discarded_waiting_round_restores_verified_prefix(self):
        decoder = DummyDecoding(make_args(algorithm="pipesd"))

        class DraftModel:
            def __init__(self):
                self.n_tokens = 99
                self.evaluated = None

            def reset(self):
                self.n_tokens = 0

            def eval(self, tokens):
                self.evaluated = list(tokens)
                self.n_tokens = len(tokens)

        decoder.draft_model = DraftModel()
        decoder._speculative_round_id = 7

        current_n_past = decoder._rollback_discarded_waiting_round([10, 11, 12])

        self.assertEqual(current_n_past, 3)
        self.assertEqual(decoder.draft_model.evaluated, [10, 11, 12])
        self.assertEqual(decoder._speculative_round_id, 8)
        self.assertTrue(decoder._is_discarded_proactive_response({
            "status": "discarded_stale_proactive_batch"
        }))

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
            self.assertEqual(engine_module.START_ENDPOINT, "http://127.0.0.1:1597/start")
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

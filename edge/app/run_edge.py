# eval_humaneval.py
import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import time
import json
import hashlib
import itertools
import platform
import socket
import statistics
import subprocess
import uuid
from pathlib import Path
from multiprocessing import Queue

from src.util import seed_everything, parse_arguments
from src.engine import Decoding
import torch.multiprocessing as mp

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

# 1. 更改类名，使其更贴切
class CloudEdgeSpeculativeEval(Decoding):
    def __init__(self, args):
        # 将队列的创建也移到这里，作为类的属性
        self.request_queue = Queue()
        self.response_queue = Queue()
        args.run_id = getattr(args, "run_id", "") or uuid.uuid4().hex
        super().__init__(args)
        self.samples = self.load_data()

    def _limit_samples_for_debug(self, samples):
        max_samples = getattr(self.args, "max_samples", None)
        if max_samples is None:
            return samples
        return samples[:max(0, max_samples)]

    def load_data(self):
        """
        从文件中加载并返回数据样本。
        """
        self.color_print(f"[Main] Loading data ({self.args.dataset}) from: {self.args.data_path}", 3)
        try:
            with open(self.args.data_path, "r", encoding="utf-8") as f:
                raw_samples = [json.loads(line) for line in f]

            dataset = getattr(self.args, "dataset", "").lower()

            if dataset == "humaneval":
                samples = self._load_humaneval_samples(raw_samples)
            elif dataset == "gsm8k":
                samples = self._load_gsm8k_samples(raw_samples)
            else:
                self.color_print(f"[Main] 未知数据集 {self.args.dataset}，默认使用 humaneval 解析格式。", 1)
                samples = self._load_humaneval_samples(raw_samples)

            samples = samples[self.start_index_of_sample:self.end_index_of_sample+1]
            samples = self._limit_samples_for_debug(samples)
            self.color_print(f"[Main] Loaded {len(samples)} samples.", 3)
            return samples
        except Exception as e:
            print(f"[Main] Error loading data: {e}")
            return [] # 如果加载失败，返回空列表以避免崩溃

    def _load_humaneval_samples(self, raw_samples):
        samples = []
        for idx, item in enumerate(raw_samples):
            prompt = item.get("prompt")
            if prompt is None:
                self.color_print(f"[Main] 跳过第 {idx} 个样本：缺少 prompt 字段", 1)
                continue
            task_id = item.get("task_id", item.get("question_id", idx))
            samples.append({
                "prompt": prompt,
                "task_id": task_id,
                "sample_index": idx,
                "canonical_solution": item.get("canonical_solution"),
                "entry_point": item.get("entry_point"),
            })
        return samples

    def _load_gsm8k_samples(self, raw_samples):
        samples = []
        for idx, item in enumerate(raw_samples):
            question = item.get("question")
            if question is None:
                self.color_print(f"[Main] 跳过第 {idx} 个样本：缺少 question 字段", 1)
                continue
            # GSM8K 不包含显式 id，这里按顺序分配 task_id
            samples.append({
                "prompt": question,
                "task_id": idx,
                "sample_index": idx,
                "reference_answer": item.get("answer"),
            })
        return samples

    def preprocess(self, input_text):
        dataset = getattr(self.args, "dataset", "").lower()
        if dataset == "mt_bench":
            turns = input_text.get("turns", [])
            # 使用首轮问题作为生成前缀，并附带类别便于区分任务
            prompt = turns[0].strip()
            if input_text.get("category"):
                prompt = f"[{input_text['category']}] {prompt}"
            raw_task_id = input_text.get("task_id", input_text.get("question_id", 0))
        elif dataset == "gsm8k":
            prompt = input_text["prompt"].strip()
            raw_task_id = input_text.get("task_id", input_text.get("question_id", 0))
        else:
            prompt = input_text['prompt'].strip()
            raw_task_id = input_text['task_id']

        try:
            task_id = int(str(raw_task_id).split('/')[-1])
        except Exception:
            task_id = raw_task_id
        task_id_offset = getattr(self.args, "task_id_offset", 0)
        if isinstance(task_id, int):
            task_id += task_id_offset
        return prompt, task_id

    def postprocess(self, input_text, output_text):
        bos_token = '<s>'
        eos_token = '</s>'
        if output_text.startswith(bos_token):
            generation = output_text[len(input_text)+len(bos_token)+1:]
        else:
            generation = output_text[len(input_text):]

        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                generation = generation[:generation.index(stop_word)].strip()

        return input_text + "\n    " + generation.replace("\t", "    ")

    def _append_bayes_record(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
        existing.append(record)
        with path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def _reset_run_duration_buffers(self):
        self._run_token_durations = []
        self._token_durations = self._run_token_durations

    def _write_latest_bayes_config(self, record: dict) -> Path:
        """Write one stable, machine-readable config consumed by eval scripts."""
        path = Path(self.exp_name) / "latest_bayes_best.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "dataset": self.args.dataset,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "run_id": self.args.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "bo_protocol": getattr(self.args, "bo_protocol", "paper"),
            "network_emulation": self._network_configuration_snapshot(),
            **record,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    def _run_latency_trial(
        self,
        thresh_single: float,
        thresh_multi: float,
        token_budget: int = None,
        tokens_per_sample: int = None,
    ):
        """Evaluate one BO candidate under the selected reproducible protocol."""
        legacy_sample_coverage = tokens_per_sample is not None
        if token_budget is None:
            token_budget = tokens_per_sample
        self._reset_run_duration_buffers()
        if hasattr(self, "dp_scheduler"):
            self.dp_scheduler.reset_workload_history()
        self.args.verify_thresh_single = thresh_single
        self.args.verify_thresh_multi = thresh_multi
        if token_budget <= 0:
            raise ValueError("BO accepted-token budget must be positive")

        protocol = "sample_coverage" if legacy_sample_coverage else getattr(self.args, "bo_protocol", "paper")
        observed_sample_indices = []
        if protocol == "paper":
            samples_iter = itertools.cycle(self.samples)
            no_progress = 0
            while len(self._token_durations) < token_budget:
                sample = next(samples_iter)
                before = len(self._token_durations)
                prompt, task_id = self.preprocess(sample)
                self._reset_state()
                self.edge_process_draft_model(
                    prompt,
                    task_id,
                    persist_result=False,
                    max_accepted_tokens=token_budget - before,
                )
                observed_sample_indices.append(sample.get("sample_index", task_id) if isinstance(sample, dict) else task_id)
                if len(self._token_durations) == before:
                    no_progress += 1
                    if no_progress >= max(2, len(self.samples)):
                        raise RuntimeError("paper BO trial could not collect accepted tokens")
                else:
                    no_progress = 0
        elif protocol == "sample_coverage":
            for sample in self.samples:
                sample_start = len(self._token_durations)
                no_progress = 0
                while len(self._token_durations) - sample_start < token_budget:
                    before = len(self._token_durations)
                    prompt, task_id = self.preprocess(sample)
                    self._reset_state()
                    remaining = token_budget - (before - sample_start)
                    self.edge_process_draft_model(
                        prompt,
                        task_id,
                        persist_result=False,
                        max_accepted_tokens=remaining,
                    )
                    observed_sample_indices.append(sample.get("sample_index", task_id) if isinstance(sample, dict) else task_id)
                    if len(self._token_durations) == before:
                        no_progress += 1
                        if no_progress >= 2:
                            raise RuntimeError(
                                f"BO trial could not collect accepted tokens from task {task_id}."
                            )
                    else:
                        no_progress = 0
        else:
            raise ValueError(f"unknown BO protocol: {protocol}")
        self._last_bo_sample_indices = observed_sample_indices
        return list(self._token_durations)

    def bayes_optimize_thresholds(self):
        if self.algorithm != "pipesd":
            raise ValueError(
                "--bayes_optimize is supported only for PipeSD; EdgeLLM "
                "follows the paper's explicit per-setup initial R1."
            )
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Bayesian optimization requires scikit-optimize (`skopt`). "
                "Install it or run without --bayes_optimize."
            ) from exc
        if not self.samples:
            self.color_print("[Main] 无样本可用于贝叶斯优化，直接退出。", 1)
            return
        if getattr(self.args, "bo_protocol", "paper") == "paper":
            tokens_target = getattr(self.args, "bayes_tokens_per_trial", 20)
        else:
            tokens_target = getattr(self.args, "bayes_tokens_per_sample", None) or 20
        log_path = Path(self.exp_name) / f"bayes_trials_run={self.args.run_id}.json"
        search_space = [
            Real(self.args.bayes_single_min, self.args.bayes_single_max, name="verify_thresh_single"),
            Real(self.args.bayes_multi_min, self.args.bayes_multi_max, name="verify_thresh_multi"),
        ]

        def objective(params):
            thresh_single, thresh_multi = params
            latencies = self._run_latency_trial(thresh_single, thresh_multi, tokens_target)
            if not latencies:
                return float("inf")
            avg_latency = float(sum(latencies) / len(latencies))
            record = {
                "thresh_single": thresh_single,
                "thresh_multi": thresh_multi,
                "avg_token_time": avg_latency,
                "num_tokens": len(latencies),
                "num_samples": len(self.samples),
                "bo_protocol": getattr(self.args, "bo_protocol", "paper"),
                "accepted_token_budget": tokens_target,
                "sample_indices": list(getattr(self, "_last_bo_sample_indices", [])),
                "run_id": self.args.run_id,
                "latencies": latencies,
            }
            self._append_bayes_record(log_path, record)
            self.color_print(
                f"[Bayes] st={thresh_single:.4f}, mt={thresh_multi:.4f}, tokens={len(latencies)}, avg={avg_latency:.6f}",
                3,
            )
            return avg_latency

        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=getattr(self.args, "bayes_calls", 16),
            n_initial_points=getattr(self.args, "bayes_init_points", 1),
            acq_func="EI",
            xi=getattr(self.args, "bayes_ei_xi", 0.1),
            random_state=self.seed,
        )
        best_single, best_multi = result.x
        best_record = {
            "best_thresh_single": best_single,
            "best_thresh_multi": best_multi,
            "best_avg_token_time": float(result.fun),
        }
        self._append_bayes_record(log_path, {"best": best_record})
        self._write_latest_bayes_config({
            **best_record,
            "accepted_token_budget": tokens_target,
            "bayes_calls": getattr(self.args, "bayes_calls", 16),
        })
        self.color_print(
            f"[Bayes] 最优阈值: st={best_single:.4f}, mt={best_multi:.4f}, avg={result.fun:.6f}",
            2,
        )
        self.args.verify_thresh_single = float(best_single)
        self.args.verify_thresh_multi = float(best_multi)
        self.verify_thresh_single = float(best_single)
        self.verify_thresh_multi = float(best_multi)
        return best_record

    @staticmethod
    def _sha256_file(path):
        if not path or not os.path.isfile(path):
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _git_value(*args):
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return None

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * percentile
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = rank - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def _build_manifest(self):
        return {
            "run_id": self.args.run_id,
            "git_commit": self._git_value("rev-parse", "HEAD"),
            "git_status": self._git_value("status", "--porcelain"),
            "created_at_unix": time.time(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "dataset": self.args.dataset,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "evaluation_protocol": self.args.evaluation_protocol,
            "target_output_tokens": self.args.target_output_tokens,
            "uplink_bandwidth_MBps": self.bandwidth_MBps,
            "downlink_bandwidth_MBps": getattr(self.args, "downlink_bandwidth_MBps", None),
            "network_shaping_mode": getattr(self.args, "network_shaping_mode", "software"),
            "network_emulation": self._network_configuration_snapshot(),
            "bo_config_path": getattr(self.args, "bo_config_path", ""),
            "bo_config_sha256": self._sha256_file(getattr(self.args, "bo_config_path", "")),
            "draft_model_path": self.args.draft_model,
            "draft_model_sha256": self._sha256_file(self.args.draft_model),
            "target_model_sha256": os.environ.get("PIPE_SD_TARGET_MODEL_SHA256"),
            "data_path": self.args.data_path,
            "data_sha256": self._sha256_file(self.args.data_path),
            "arguments": vars(self.args),
        }

    def _build_run_summary(self, sample_results):
        actual_tokens = sum(int(item.get("output_length", 0)) for item in sample_results)
        total_time = sum(float(item.get("total_time", 0.0)) for item in sample_results)
        durations = [
            float(duration)
            for item in sample_results
            for duration in item.get("token_durations", [])
        ]
        ttft = [
            float(item["time_to_first_token_seconds"])
            for item in sample_results
            if item.get("time_to_first_token_seconds") is not None
        ]
        num_verifications = sum(
            int(item.get("verify_stats", {}).get("num_verifications", 0))
            for item in sample_results
        )
        draft_tokens = sum(
            sum(item.get("verify_spec_lengths", []))
            for item in sample_results
        )
        accepted_drafts = sum(
            sum(item.get("verify_accept_lengths", []))
            for item in sample_results
        )
        rollback_events = sum(
            int(item.get("diagnostics", {}).get("rollback_events", 0))
            for item in sample_results
        )
        batch_sizes = [
            int(batch.get("actual_batch_size", 0))
            for item in sample_results
            for batch in item.get("batch_trace", [])
            if int(batch.get("actual_batch_size", 0)) > 0
        ]
        measured_energy = [
            float(
                item.get("model_energy_joules")
                if item.get("model_energy_joules") is not None
                else item["gpu_power_integral_joules"]
            )
            for item in sample_results
            if (
                item.get("model_energy_joules") is not None
                or item.get("gpu_power_integral_joules") is not None
            )
        ]
        measured_prefill_energy = [
            float(item["prompt_prefill_gpu_energy_joules"])
            for item in sample_results
            if item.get("prompt_prefill_gpu_energy_joules") is not None
        ]
        measured_nav_energy = [
            float(item["nav_gpu_energy_joules"])
            for item in sample_results
            if item.get("nav_gpu_energy_joules") is not None
        ]
        measured_energy_durations = [
            float(item["energy_measurement_duration_seconds"])
            for item in sample_results
            if item.get("energy_measurement_duration_seconds") is not None
        ]
        total_energy = (
            sum(measured_energy)
            if len(measured_energy) == len(sample_results)
            else None
        )
        total_prefill_energy = (
            sum(measured_prefill_energy)
            if len(measured_prefill_energy) == len(sample_results)
            else None
        )
        total_nav_energy = (
            sum(measured_nav_energy)
            if len(measured_nav_energy) == len(sample_results)
            else None
        )
        total_energy_duration = (
            sum(measured_energy_durations)
            if len(measured_energy_durations) == len(sample_results)
            else None
        )
        energy_scopes = {
            item.get("energy_scope") for item in sample_results if item.get("energy_scope")
        }
        energy_sources = {
            item.get("energy_source") for item in sample_results if item.get("energy_source")
        }
        return {
            "evaluation_protocol": self.args.evaluation_protocol,
            # Keep target_output_tokens for command/result compatibility.  Under
            # paper_table1 it is now the requested cloud-accepted draft-token
            # count; actual_output_tokens still describes committed output and
            # therefore also includes the target model's extra token per NAV.
            "target_output_tokens": int(self.args.target_output_tokens),
            "target_accepted_draft_tokens": int(self.args.target_output_tokens),
            "actual_accepted_draft_tokens": accepted_drafts,
            "stopping_criterion": "cloud_accepted_draft_tokens",
            "actual_output_tokens": actual_tokens,
            "sample_indices": [item.get("sample_index") for item in sample_results],
            "num_samples": len(sample_results),
            "total_time_seconds": total_time,
            # The paper benchmark unit is a cloud-accepted draft token.  Keep
            # explicit output-token metrics because every NAV may also commit
            # one target-model final token.
            "tpt_normalization_token_type": "cloud_accepted_draft_tokens",
            "weighted_tpt_seconds": (
                total_time / accepted_drafts if accepted_drafts else None
            ),
            "weighted_tpt_ms": (
                1000.0 * total_time / accepted_drafts if accepted_drafts else None
            ),
            "accepted_token_tpt_seconds": (
                total_time / accepted_drafts if accepted_drafts else None
            ),
            "accepted_token_tpt_ms": (
                1000.0 * total_time / accepted_drafts if accepted_drafts else None
            ),
            "output_token_tpt_seconds": (
                total_time / actual_tokens if actual_tokens else None
            ),
            "output_token_tpt_ms": (
                1000.0 * total_time / actual_tokens if actual_tokens else None
            ),
            "throughput_tokens_per_second": (
                accepted_drafts / total_time if total_time else None
            ),
            "accepted_tokens_per_second": (
                accepted_drafts / total_time if total_time else None
            ),
            "output_tokens_per_second": (
                actual_tokens / total_time if total_time else None
            ),
            "token_latency_p50_seconds": self._percentile(durations, 0.50),
            "token_latency_p95_seconds": self._percentile(durations, 0.95),
            "token_latency_p99_seconds": self._percentile(durations, 0.99),
            "mean_ttft_seconds": statistics.fmean(ttft) if ttft else None,
            "ttft_p95_seconds": self._percentile(ttft, 0.95),
            "num_verifications": num_verifications,
            "verification_frequency": (
                num_verifications / accepted_drafts if accepted_drafts else None
            ),
            "verification_frequency_per_output_token": (
                num_verifications / actual_tokens if actual_tokens else None
            ),
            "mean_draft_length": draft_tokens / num_verifications if num_verifications else None,
            "acceptance_rate": accepted_drafts / draft_tokens if draft_tokens else None,
            "rollback_rate": rollback_events / num_verifications if num_verifications else None,
            "mean_actual_batch_size": statistics.fmean(batch_sizes) if batch_sizes else None,
            "cap_hit_count": sum(1 for item in sample_results if item.get("generation_cap_hit")),
            "cap_hit_rate": (
                sum(1 for item in sample_results if item.get("generation_cap_hit")) / len(sample_results)
                if sample_results else None
            ),
            "eos_count": sum(1 for item in sample_results if item.get("ended_with_eos")),
            "model_energy_joules": total_energy,
            "gpu_energy_joules": total_energy,
            "prompt_prefill_gpu_energy_joules": total_prefill_energy,
            "nav_gpu_energy_joules": total_nav_energy,
            "energy_measurement_duration_seconds": total_energy_duration,
            "average_active_compute_gpu_power_watts": (
                total_energy / total_energy_duration
                if total_energy is not None and total_energy_duration
                else None
            ),
            "nav_energy_measurement_count": sum(
                len(item.get("nav_energy_trace", [])) for item in sample_results
            ),
            "energy_scope": (
                next(iter(energy_scopes)) if len(energy_scopes) == 1 else sorted(energy_scopes)
            ),
            "energy_source": (
                next(iter(energy_sources)) if len(energy_sources) == 1 else sorted(energy_sources)
            ),
            "energy_normalization_token_type": "cloud_accepted_draft_tokens",
            "energy_included_stages": [
                "cloud_prompt_prefill",
                "target_model_nav_compute",
            ],
            "energy_excluded_stages": [
                "between_nav_gpu_idle",
                "edge_draft_wait",
                "network_transfer",
                "proactive_wait_and_transfer",
                "model_load",
                "model_state_restore_and_save",
            ],
            "gpu_energy_joules_per_100_tokens": (
                100.0 * total_energy / accepted_drafts
                if total_energy is not None and accepted_drafts
                else None
            ),
            "gpu_energy_joules_per_100_accepted_tokens": (
                100.0 * total_energy / accepted_drafts
                if total_energy is not None and accepted_drafts
                else None
            ),
            "model_energy_joules_per_100_tokens": (
                100.0 * total_energy / accepted_drafts
                if total_energy is not None and accepted_drafts
                else None
            ),
            "gpu_energy_joules_per_100_output_tokens": (
                100.0 * total_energy / actual_tokens
                if total_energy is not None and actual_tokens
                else None
            ),
        }

    def _paper_result_path(self):
        bandwidth_label = f"{self.bandwidth_MBps:g}"
        legacy_path = Path(self.exp2path(bandwidth_label))
        safe_run_id = "".join(ch for ch in self.args.run_id if ch.isalnum() or ch in "-_")
        return legacy_path.with_name(f"{legacy_path.stem}_run={safe_run_id}.json")

    def _run_paper_table1(self):
        # Preserve the existing CLI option for compatibility, but interpret the
        # paper_table1 target as cloud-accepted draft tokens rather than total
        # committed output tokens.
        target_accepted_tokens = int(self.args.target_output_tokens)
        if target_accepted_tokens <= 0:
            raise ValueError("target_output_tokens must be positive")
        if not self.samples:
            raise RuntimeError("no dataset samples are available")

        self._reset_run_duration_buffers()
        sample_results = []
        samples_iter = itertools.cycle(self.samples)
        actual_accepted_tokens = 0
        no_progress = 0
        while actual_accepted_tokens < target_accepted_tokens:
            sample = next(samples_iter)
            prompt, task_id = self.preprocess(sample)
            self._reset_state()
            remaining_accepted_tokens = target_accepted_tokens - actual_accepted_tokens
            result = self.edge_process_draft_model(
                prompt,
                task_id,
                persist_result=False,
                max_cloud_accepted_tokens=remaining_accepted_tokens,
            )
            if not isinstance(result, dict):
                no_progress += 1
                if no_progress >= len(self.samples):
                    raise RuntimeError("a complete dataset pass produced no accepted tokens")
                continue

            produced = int(result.get("output_length", 0))
            accepted = sum(
                int(value) for value in result.get("verify_accept_lengths", [])
            )
            if accepted < 0:
                raise RuntimeError("cloud returned a negative accepted-token count")
            if accepted > remaining_accepted_tokens:
                raise RuntimeError(
                    "cloud accepted-token count exceeded the remaining paper_table1 "
                    f"budget: accepted={accepted}, remaining={remaining_accepted_tokens}"
                )
            if accepted > produced:
                raise RuntimeError(
                    "cloud accepted more draft tokens than the committed output contains: "
                    f"accepted={accepted}, output_length={produced}"
                )

            result["accepted_draft_tokens"] = accepted
            result["cumulative_accepted_draft_tokens"] = actual_accepted_tokens + accepted
            result["sample_index"] = sample.get("sample_index", task_id)
            result["dataset_task_id"] = sample.get("task_id", task_id)
            if self.args.dataset.lower() == "gsm8k":
                result["reference_answer"] = sample.get("reference_answer")
            else:
                result["canonical_solution"] = sample.get("canonical_solution")
                result["entry_point"] = sample.get("entry_point")
            sample_results.append(result)

            if accepted <= 0:
                no_progress += 1
                if no_progress >= len(self.samples):
                    raise RuntimeError("a complete dataset pass produced no accepted tokens")
                continue
            no_progress = 0
            actual_accepted_tokens += accepted

        if actual_accepted_tokens != target_accepted_tokens:
            raise RuntimeError(
                "paper protocol reached "
                f"{actual_accepted_tokens} cloud-accepted draft tokens, "
                f"expected {target_accepted_tokens}"
            )

        payload = {
            "manifest": self._build_manifest(),
            "summary": self._build_run_summary(sample_results),
            "samples": sample_results,
        }
        result_path = self._paper_result_path()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self.color_print(f"[Main] paper_table1 result: {result_path}", 2)
        return payload

    @torch.no_grad()
    def eval(self):
        if getattr(self.args, "bayes_optimize", False):
            if self.algorithm != "edgeLLM" and self.verify_strategy != "hybrid":
                self.color_print("[Main] 当前策略非 hybrid，将自动切换为 hybrid 进行阈值搜索。", 3)
                self.verify_strategy = "hybrid"
            self.bayes_optimize_thresholds()
            if getattr(self.args, "bayes_only", False):
                self.color_print("[Main] BO 完成，已按 --bayes_only 跳过正式评测。", 2)
                return
            self.color_print("[Main] BO 完成，使用最优阈值继续正式评测。", 2)
        # start_time = time.time()
        seed_everything(self.args.seed)
        if getattr(self.args, "evaluation_protocol", "sample_index") == "paper_table1":
            return self._run_paper_table1()
        self._reset_run_duration_buffers()
        for i in range(1):
            for sample in self.samples:
                prompt, task_id = self.preprocess(sample)
                self._reset_state()
                bandwidth_label = f"{self.bandwidth_MBps:g}"
                # bandwidth_label = "{0}"
                path = self.exp2path(bandwidth_label)

                if os.path.exists(path):
                    print('path exists:', path)
                    with open(path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            data = []
                    if any(entry['task_id'] == task_id for entry in data):
                        self.color_print(f"[Main] Task {task_id} already evaluated. Skipping.", 3)
                        continue
                    
                
                self.edge_process_draft_model(prompt, task_id)
        

if __name__ == "__main__":
    MAX_RETRY = 5
    for i in range(MAX_RETRY):
        try:
            args = parse_arguments()
            evaluator = CloudEdgeSpeculativeEval(args)
            evaluator.eval()
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            if i == MAX_RETRY - 1:
                raise
            time.sleep(2)

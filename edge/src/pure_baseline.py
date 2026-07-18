"""Standalone pure-edge and pure-cloud autoregressive baselines.

This module intentionally does not depend on :mod:`src.engine`.  The two pure
deployment baselines must not initialize the speculative-decoding transport,
create a cloud task, or execute NAV. Keeping the implementation separate also
makes their timing boundary explicit: TPT covers the complete warm-model local
request while prompt prefill and decode are retained as diagnostic sub-stages.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - depends on the deployment host
    pynvml = None


DEFAULT_MODELS = {
    "pure_edge": {
        "humaneval": "pre_models/deepseek-coder-1.3b-instruct-GGUF/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
        "gsm8k": "pre_models/tinyllama-1.1b-chat-v1.0-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    },
    "pure_cloud": {
        "humaneval": "../cloud/pre_models/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "gsm8k": "../cloud/pre_models/Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
    },
}

MODE_RUNTIME_DEFAULTS = {
    # Match the Edge draft-model defaults in edge/src/util.py.
    "pure_edge": {
        "seed": 3407,
        "threads": 1,
        "ctx_size": 1024,
        "temp": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    },
    # Match the Cloud target-model defaults in cloud/src/util.py.
    "pure_cloud": {
        "seed": 3407,
        "threads": 1,
        "ctx_size": 1024,
        "temp": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    },
}


def resolve_runtime_defaults(args) -> None:
    """Fill unspecified pure-baseline settings from the matching model role."""
    defaults = MODE_RUNTIME_DEFAULTS[args.mode]
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = int(math.floor(rank))
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def load_samples(
    data_path: Path,
    dataset: str,
    start_index: int,
    end_index: int,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with data_path.open("r", encoding="utf-8") as handle:
        raw_samples = [json.loads(line) for line in handle if line.strip()]

    samples: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_samples):
        if dataset == "gsm8k":
            prompt = item.get("question")
            if prompt is None:
                continue
            samples.append(
                {
                    "prompt": str(prompt).strip(),
                    "task_id": item.get("task_id", index),
                    "sample_index": index,
                    "reference_answer": item.get("answer"),
                }
            )
        elif dataset == "humaneval":
            prompt = item.get("prompt")
            if prompt is None:
                continue
            samples.append(
                {
                    "prompt": str(prompt).strip(),
                    "task_id": item.get("task_id", item.get("question_id", index)),
                    "sample_index": index,
                    "canonical_solution": item.get("canonical_solution"),
                    "entry_point": item.get("entry_point"),
                }
            )
        else:
            raise ValueError(f"unsupported dataset: {dataset}")

    if end_index < start_index:
        raise ValueError("end_index_of_sample must be >= start_index_of_sample")
    selected = samples[start_index : end_index + 1]
    return selected if max_samples is None else selected[: max(0, max_samples)]


class NVMLPowerMeter:
    """Integrate GPU board power during a measured model-compute interval."""

    source = "nvml_gpu_board_power"

    def __init__(self, device_index: int = 0, sample_interval: float = 0.005):
        self.sample_interval = max(0.001, float(sample_interval))
        self.handle = None
        self.last_measurement_duration_seconds: Optional[float] = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                pynvml.nvmlDeviceGetPowerUsage(self.handle)
            except Exception:
                self.handle = None

    @property
    def available(self) -> bool:
        return self.handle is not None

    def measure(self, function):
        if not self.available:
            return function(), None
        samples: List[Tuple[float, float]] = []
        stop_event = threading.Event()
        def read_power() -> None:
            try:
                samples.append(
                    (time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
                )
            except Exception:
                pass

        def sampler() -> None:
            while not stop_event.wait(self.sample_interval):
                read_power()

        read_power()
        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        try:
            result = function()
        finally:
            read_power()
            stop_event.set()
            thread.join()

        self.last_measurement_duration_seconds = (
            samples[-1][0] - samples[0][0] if len(samples) >= 2 else None
        )

        energy = sum(
            (power_a + power_b) * 0.5 * (time_b - time_a)
            for (time_a, power_a), (time_b, power_b) in zip(samples, samples[1:])
            if time_b > time_a
        )
        return result, energy if len(samples) >= 2 else None


class NullEnergyMeter:
    source = "unavailable"
    available = False
    last_measurement_duration_seconds = None

    @staticmethod
    def measure(function):
        return function(), None


def select_energy_meter(
    mode: str,
    gpu_device: int,
    sample_interval: float,
    n_gpu_layers: int = 0,
):
    if mode == "pure_cloud":
        meter = NVMLPowerMeter(gpu_device, sample_interval)
        return meter if meter.available else NullEnergyMeter()
    # Pure Edge energy is intentionally not measured.  RAPL commonly requires
    # privileges that are unavailable on the evaluation edge host, and silently
    # mixing partial CPU/GPU scopes would make the four-mode table misleading.
    return NullEnergyMeter()


@dataclass
class GenerationConfig:
    max_generated_tokens: int
    top_k: int
    top_p: float
    temperature: float


def generate_one(
    model,
    prompt: str,
    token_budget: int,
    config: GenerationConfig,
    energy_meter,
) -> Dict[str, Any]:
    """Run one local autoregressive sample.

    TPT covers warm-model request processing end to end (reset/tokenization,
    prompt prefill, decode, and detokenization). Model energy, when enabled for
    Pure Cloud, continuously covers prompt prefill plus complete decode.
    """
    preprocess_started = time.perf_counter()
    model.reset()
    prefix_tokens = model.tokenize(prompt.encode("utf-8"), add_bos=True)
    request_preprocess_time = time.perf_counter() - preprocess_started
    generated_tokens: List[int] = []
    durations: List[float] = []
    prefill_time = 0.0
    decode_time = 0.0

    def prefill_compute() -> None:
        nonlocal prefill_time
        prefill_started = time.perf_counter()
        model.eval(prefix_tokens)
        prefill_time = time.perf_counter() - prefill_started

    _, prompt_prefill_energy_joules = energy_meter.measure(prefill_compute)
    prompt_prefill_energy_duration = getattr(
        energy_meter, "last_measurement_duration_seconds", None
    )

    def decode_compute() -> None:
        nonlocal decode_time
        started = time.perf_counter()
        limit = min(config.max_generated_tokens, max(0, int(token_budget)))
        while len(generated_tokens) < limit:
            step_started = time.perf_counter()
            token = int(
                model.sample(
                    top_k=config.top_k,
                    top_p=config.top_p,
                    temp=config.temperature,
                )
            )
            model.eval([token])
            durations.append(time.perf_counter() - step_started)
            generated_tokens.append(token)
            if token == model.token_eos():
                break
        decode_time = time.perf_counter() - started

    _, decode_energy_joules = energy_meter.measure(decode_compute)
    decode_energy_duration = getattr(
        energy_meter, "last_measurement_duration_seconds", None
    )
    energy_joules = (
        float(prompt_prefill_energy_joules) + float(decode_energy_joules)
        if prompt_prefill_energy_joules is not None
        and decode_energy_joules is not None
        else None
    )
    energy_duration = (
        float(prompt_prefill_energy_duration) + float(decode_energy_duration)
        if prompt_prefill_energy_duration is not None
        and decode_energy_duration is not None
        else None
    )
    postprocess_started = time.perf_counter()
    continuation = model.detokenize(generated_tokens).decode("utf-8", "ignore")
    full_text = model.detokenize(prefix_tokens + generated_tokens).decode("utf-8", "ignore")
    output_postprocess_time = time.perf_counter() - postprocess_started
    end_to_end_time = (
        request_preprocess_time
        + prefill_time
        + decode_time
        + output_postprocess_time
    )
    ended_with_eos = bool(generated_tokens and generated_tokens[-1] == model.token_eos())
    return {
        "output_length": len(generated_tokens),
        "total_time": end_to_end_time,
        "end_to_end_time_seconds": end_to_end_time,
        "request_preprocess_time_seconds": request_preprocess_time,
        "prefill_time_seconds": prefill_time,
        "decode_time_seconds": decode_time,
        "output_postprocess_time_seconds": output_postprocess_time,
        "time_to_first_token_seconds": (
            request_preprocess_time + prefill_time + durations[0]
            if durations
            else None
        ),
        "token_durations": durations,
        "output": full_text,
        "generated_text": continuation,
        "processed_output": continuation,
        "requested_output_tokens": min(config.max_generated_tokens, token_budget),
        "ended_with_eos": ended_with_eos,
        "generation_cap_hit": bool(
            generated_tokens
            and len(generated_tokens) == min(config.max_generated_tokens, token_budget)
            and not ended_with_eos
        ),
        "model_energy_joules": energy_joules,
        "prompt_prefill_gpu_energy_joules": prompt_prefill_energy_joules,
        "decode_gpu_energy_joules": decode_energy_joules,
        "energy_measurement_duration_seconds": energy_duration,
        "prompt_prefill_energy_measurement_duration_seconds": (
            prompt_prefill_energy_duration
        ),
        "decode_energy_measurement_duration_seconds": decode_energy_duration,
        "energy_scope": "model_prompt_prefill_plus_autoregressive_decode",
        "energy_included_stages": ["prompt_prefill", "autoregressive_decode"],
        "energy_excluded_stages": [
            "model_load",
            "prompt_tokenization",
            "output_detokenization",
        ],
        "energy_source": energy_meter.source,
        "verify_stats": {"num_verifications": 0},
        "verify_spec_lengths": [],
        "verify_accept_lengths": [],
        "diagnostics": {},
        "batch_trace": [],
    }


def build_summary(
    mode: str,
    target_tokens: int,
    sample_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    actual_tokens = sum(int(sample["output_length"]) for sample in sample_results)
    total_time = sum(float(sample["total_time"]) for sample in sample_results)
    durations = [
        float(value)
        for sample in sample_results
        for value in sample.get("token_durations", [])
    ]
    ttft = [
        float(sample["time_to_first_token_seconds"])
        for sample in sample_results
        if sample.get("time_to_first_token_seconds") is not None
    ]
    measured_energy = [
        float(sample["model_energy_joules"])
        for sample in sample_results
        if sample.get("model_energy_joules") is not None
    ]
    energy = sum(measured_energy) if len(measured_energy) == len(sample_results) else None
    measured_prefill_energy = [
        float(sample["prompt_prefill_gpu_energy_joules"])
        for sample in sample_results
        if sample.get("prompt_prefill_gpu_energy_joules") is not None
    ]
    prompt_prefill_energy = (
        sum(measured_prefill_energy)
        if len(measured_prefill_energy) == len(sample_results)
        else None
    )
    measured_decode_energy = [
        float(sample["decode_gpu_energy_joules"])
        for sample in sample_results
        if sample.get("decode_gpu_energy_joules") is not None
    ]
    decode_energy = (
        sum(measured_decode_energy)
        if len(measured_decode_energy) == len(sample_results)
        else None
    )
    measured_prefill_energy_durations = [
        float(sample["prompt_prefill_energy_measurement_duration_seconds"])
        for sample in sample_results
        if sample.get("prompt_prefill_energy_measurement_duration_seconds")
        is not None
    ]
    prompt_prefill_energy_duration = (
        sum(measured_prefill_energy_durations)
        if len(measured_prefill_energy_durations) == len(sample_results)
        else None
    )
    measured_decode_energy_durations = [
        float(sample["decode_energy_measurement_duration_seconds"])
        for sample in sample_results
        if sample.get("decode_energy_measurement_duration_seconds") is not None
    ]
    decode_energy_duration = (
        sum(measured_decode_energy_durations)
        if len(measured_decode_energy_durations) == len(sample_results)
        else None
    )
    measured_energy_durations = [
        float(sample["energy_measurement_duration_seconds"])
        for sample in sample_results
        if sample.get("energy_measurement_duration_seconds") is not None
    ]
    energy_duration = (
        sum(measured_energy_durations)
        if len(measured_energy_durations) == len(sample_results)
        else None
    )
    sources = sorted({sample.get("energy_source", "unavailable") for sample in sample_results})
    if mode == "pure_edge":
        energy = None
        prompt_prefill_energy = None
        decode_energy = None
        prompt_prefill_energy_duration = None
        decode_energy_duration = None
        energy_duration = None
        energy_scope = "not_measured_no_rapl_permission"
        energy_source: Any = "not_measured"
        energy_included_stages: List[str] = []
        energy_excluded_stages = ["all_pure_edge_energy"]
    else:
        energy_scope = "cloud_gpu_prompt_prefill_plus_autoregressive_decode"
        energy_source = sources[0] if len(sources) == 1 else sources
        energy_included_stages = ["prompt_prefill", "autoregressive_decode"]
        energy_excluded_stages = [
            "model_load",
            "prompt_tokenization",
            "output_detokenization",
        ]
    summary = {
        "evaluation_protocol": "paper_table1",
        "target_output_tokens": int(target_tokens),
        "actual_output_tokens": actual_tokens,
        "sample_indices": [sample.get("sample_index") for sample in sample_results],
        "num_samples": len(sample_results),
        "total_time_seconds": total_time,
        "timing_scope": "warm_model_request_end_to_end_excluding_model_load",
        "tpt_normalization_token_type": "committed_output_tokens",
        "weighted_tpt_seconds": total_time / actual_tokens if actual_tokens else None,
        "weighted_tpt_ms": 1000.0 * total_time / actual_tokens if actual_tokens else None,
        "throughput_tokens_per_second": actual_tokens / total_time if total_time else None,
        "token_latency_p50_seconds": percentile(durations, 0.50),
        "token_latency_p95_seconds": percentile(durations, 0.95),
        "token_latency_p99_seconds": percentile(durations, 0.99),
        "mean_ttft_seconds": statistics.fmean(ttft) if ttft else None,
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "total_prefill_time_seconds": sum(
            float(sample.get("prefill_time_seconds", 0.0)) for sample in sample_results
        ),
        "total_decode_time_seconds": sum(
            float(sample.get("decode_time_seconds", 0.0)) for sample in sample_results
        ),
        "total_request_preprocess_time_seconds": sum(
            float(sample.get("request_preprocess_time_seconds", 0.0))
            for sample in sample_results
        ),
        "total_output_postprocess_time_seconds": sum(
            float(sample.get("output_postprocess_time_seconds", 0.0))
            for sample in sample_results
        ),
        "num_verifications": None,
        "verification_frequency": None,
        "mean_draft_length": None,
        "acceptance_rate": None,
        "rollback_rate": None,
        "mean_actual_batch_size": None,
        "cap_hit_count": sum(bool(sample.get("generation_cap_hit")) for sample in sample_results),
        "cap_hit_rate": (
            sum(bool(sample.get("generation_cap_hit")) for sample in sample_results)
            / len(sample_results)
            if sample_results
            else None
        ),
        "eos_count": sum(bool(sample.get("ended_with_eos")) for sample in sample_results),
        "model_energy_joules": energy,
        "prompt_prefill_gpu_energy_joules": prompt_prefill_energy,
        "decode_gpu_energy_joules": decode_energy,
        "prompt_prefill_energy_measurement_duration_seconds": (
            prompt_prefill_energy_duration
        ),
        "decode_energy_measurement_duration_seconds": decode_energy_duration,
        "model_energy_joules_per_100_tokens": (
            100.0 * energy / actual_tokens if energy is not None and actual_tokens else None
        ),
        "energy_measurement_duration_seconds": energy_duration,
        "average_model_compute_power_watts": (
            energy / energy_duration
            if energy is not None and energy_duration
            else None
        ),
        "energy_scope": energy_scope,
        "energy_source": energy_source,
        "energy_normalization_token_type": "committed_output_tokens",
        "energy_included_stages": energy_included_stages,
        "energy_excluded_stages": energy_excluded_stages,
        "gpu_energy_joules": energy if mode == "pure_cloud" else None,
        "gpu_energy_joules_per_100_tokens": (
            100.0 * energy / actual_tokens
            if mode == "pure_cloud" and energy is not None and actual_tokens
            else None
        ),
        "edge_energy_joules": None,
        "edge_energy_joules_per_100_tokens": None,
    }
    return summary


class PureBaselineEvaluator:
    def __init__(self, args, model_factory=None, energy_meter=None):
        self.args = args
        self.mode = args.mode
        resolve_runtime_defaults(args)
        self.dataset = args.dataset.lower()
        self.run_id = args.run_id or uuid.uuid4().hex
        self.data_path = Path(args.data_path or f"data/{self.dataset}.jsonl")
        default_model = DEFAULT_MODELS[self.mode][self.dataset]
        self.model_path = Path(args.model_path or default_model)
        self.samples = load_samples(
            self.data_path,
            self.dataset,
            args.start_index_of_sample,
            args.end_index_of_sample,
            args.max_samples,
        )
        if not self.samples:
            raise RuntimeError("no dataset samples are available")

        random.seed(args.seed)
        np.random.seed(args.seed)
        if model_factory is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError("llama-cpp-python is required for pure baselines") from exc
            model_factory = Llama
        n_gpu_layers = args.n_gpu_layers
        if n_gpu_layers is None:
            n_gpu_layers = -1 if self.mode == "pure_cloud" else 0
        model_started = time.time()
        self.model = model_factory(
            model_path=str(self.model_path),
            n_gpu_layers=n_gpu_layers,
            n_threads=args.threads,
            n_threads_batch=args.threads,
            verbose=False,
            logits_all=True,
            n_ctx=args.ctx_size,
            seed=args.seed,
        )
        self.model_load_seconds = time.time() - model_started
        self.n_gpu_layers = n_gpu_layers
        self.energy_meter = energy_meter or select_energy_meter(
            self.mode,
            args.gpu_energy_device,
            args.energy_sample_interval,
            self.n_gpu_layers,
        )

    def _manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--porcelain"),
            "created_at_unix": time.time(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "dataset": self.dataset,
            "algorithm": self.mode,
            "deployment_mode": self.mode,
            "seed": self.args.seed,
            "evaluation_protocol": "paper_table1",
            "target_output_tokens": self.args.target_output_tokens,
            "timing_scope": "warm_model_request_end_to_end_excluding_model_load",
            "energy_scope": (
                "cloud_gpu_prompt_prefill_plus_autoregressive_decode"
                if self.mode == "pure_cloud"
                else "not_measured_no_rapl_permission"
            ),
            "network_included": False,
            "model_path": str(self.model_path),
            "model_sha256": sha256_file(self.model_path),
            "model_n_gpu_layers": self.n_gpu_layers,
            "model_load_seconds": self.model_load_seconds,
            "data_path": str(self.data_path),
            "data_sha256": sha256_file(self.data_path),
            "result_tag": self.args.result_tag,
            "arguments": vars(self.args),
        }

    def _result_path(self) -> Path:
        safe_run_id = "".join(ch for ch in self.run_id if ch.isalnum() or ch in "-_")
        tag = f"_tag={self.args.result_tag}" if self.args.result_tag else ""
        return (
            Path(self.args.output_root)
            / self.dataset
            / self.mode
            / f"{self.mode}{tag}_run={safe_run_id}.json"
        )

    def run(self) -> Tuple[Dict[str, Any], Path]:
        target_tokens = int(self.args.target_output_tokens)
        if target_tokens <= 0:
            raise ValueError("target_output_tokens must be positive")
        config = GenerationConfig(
            max_generated_tokens=int(self.args.max_generated_tokens),
            top_k=int(self.args.top_k),
            top_p=float(self.args.top_p),
            temperature=float(self.args.temp),
        )
        sample_results: List[Dict[str, Any]] = []
        produced = 0
        no_progress = 0
        for sample in itertools.cycle(self.samples):
            if produced >= target_tokens:
                break
            result = generate_one(
                self.model,
                sample["prompt"],
                target_tokens - produced,
                config,
                self.energy_meter,
            )
            if result["output_length"] <= 0:
                no_progress += 1
                if no_progress >= len(self.samples):
                    raise RuntimeError("a complete dataset pass produced no tokens")
                continue
            no_progress = 0
            result["task_id"] = sample["task_id"]
            result["sample_index"] = sample["sample_index"]
            result["dataset_task_id"] = sample["task_id"]
            for key in ("reference_answer", "canonical_solution", "entry_point"):
                if key in sample:
                    result[key] = sample[key]
            sample_results.append(result)
            produced += int(result["output_length"])

        payload = {
            "manifest": self._manifest(),
            "summary": build_summary(self.mode, target_tokens, sample_results),
            "samples": sample_results,
        }
        result_path = self._result_path()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload, result_path

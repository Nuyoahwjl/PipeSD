"""Local fixed-window speculative decoding for pure-cloud and pure-edge modes.

The collaborative ``vanilla`` path places the draft model at the edge and the
target model in the cloud.  This module keeps the same fixed-window proposal,
prefix acceptance, synchronous verification, and accepted-draft-token budget,
but co-locates both models and replaces the WAN request with an in-process
handoff.  It intentionally does not implement PipeSD overlap.
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import random
import socket
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pure_baseline import (
    NullEnergyMeter,
    git_value,
    load_samples,
    percentile,
    select_energy_meter,
    sha256_file,
)


DEFAULT_MODEL_PAIRS = {
    "humaneval": {
        "draft": "pre_models/deepseek-coder-1.3b-instruct-GGUF/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
        "target": "../cloud/pre_models/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
    },
    "gsm8k": {
        "draft": "pre_models/tinyllama-1.1b-chat-v1.0-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "target": "../cloud/pre_models/Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
    },
}


def softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = array - np.max(array)
    exponent = np.exp(array)
    return exponent / np.sum(exponent)


def positive_distribution(values: Sequence[float]) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.shape, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total


def _measure(meter, function) -> Tuple[Any, Optional[float], Optional[float]]:
    result, energy = meter.measure(function)
    duration = getattr(meter, "last_measurement_duration_seconds", None)
    return result, energy, duration


def verify_local_proposal(
    target_model,
    speculative_tokens: Sequence[int],
    draft_probabilities: Sequence[np.ndarray],
    confirmed_length: int,
    *,
    seed: int,
    top_k: int,
    top_p: float,
    temperature: float,
) -> Dict[str, Any]:
    """Verify one proposal with the same acceptance rule as cloud Vanilla SD."""
    if not speculative_tokens:
        raise ValueError("a local verification proposal must contain at least one token")

    target_model.eval(list(speculative_tokens))
    count = len(speculative_tokens)
    target_scores = np.asarray(
        target_model.scores[confirmed_length - 1 : confirmed_length - 1 + count],
        dtype=np.float64,
    )
    if target_scores.shape[0] != count:
        raise RuntimeError(
            "target model did not expose one logit row per speculative token: "
            f"expected={count}, actual={target_scores.shape[0]}"
        )
    target_probabilities = np.stack([softmax(row) for row in target_scores])
    draft_matrix = np.stack(draft_probabilities)
    indices = np.arange(count)
    target_token_probabilities = target_probabilities[indices, speculative_tokens]
    draft_token_probabilities = draft_matrix[indices, speculative_tokens]
    ratios = target_token_probabilities / (draft_token_probabilities + 1e-9)

    accepted = 0
    random_trace = []
    for index, ratio in enumerate(ratios):
        position = confirmed_length + index
        random_value = random.Random(seed + position).random()
        random_trace.append(
            {"position": position, "random": random_value, "ratio": float(ratio)}
        )
        if ratio >= 1.0 or random_value < float(ratio):
            accepted += 1
        else:
            break

    # Discard the unaccepted suffix before adding the target continuation.
    target_model.n_tokens = confirmed_length + accepted
    if accepted < count:
        difference = target_probabilities[accepted] - draft_matrix[accepted]
        final_token = int(
            np.random.default_rng(seed).choice(
                difference.size, p=positive_distribution(difference)
            )
        )
    else:
        final_token = int(
            target_model.sample(top_k=top_k, top_p=top_p, temp=temperature)
        )

    # The cloud implementation carries this token as ``final_token`` and feeds
    # it before the next NAV.  Evaluating it immediately is state-equivalent for
    # this synchronous, non-overlapped local path.
    target_model.eval([final_token])
    return {
        "n_accepted": accepted,
        "n_speculative": count,
        "final_token": final_token,
        "acceptance_ratios": [float(value) for value in ratios],
        "random_trace": random_trace,
    }


def generate_one_local_sd(
    draft_model,
    target_model,
    prompt: str,
    accepted_token_budget: int,
    *,
    verify_num: int,
    max_generated_tokens: int,
    seed: int,
    top_k: int,
    top_p: float,
    temperature: float,
    energy_meter=None,
) -> Dict[str, Any]:
    """Run one synchronous local speculative-decoding request.

    Decode timing starts after both prompt prefills, matching the current
    collaborative engine's measured boundary.  The accepted-token budget only
    counts draft tokens accepted by the target; target continuation tokens are
    committed output but do not consume that budget.
    """
    if verify_num <= 0:
        raise ValueError("verify_num must be positive")
    if accepted_token_budget <= 0:
        raise ValueError("accepted_token_budget must be positive")
    energy_meter = energy_meter or NullEnergyMeter()

    preprocess_started = time.perf_counter()
    draft_model.reset()
    target_model.reset()
    # Match the existing edge-cloud Vanilla protocol: the edge draft tokenizer
    # is the canonical tokenizer, and the cloud target receives those token IDs
    # directly in the /init payload.  Re-tokenizing the raw prompt with the
    # target GGUF can differ because of BOS/add-space tokenizer metadata even
    # when the two models share the token-ID vocabulary required by SD.
    prefix_tokens = list(draft_model.tokenize(prompt.encode("utf-8"), add_bos=True))
    preprocess_time = time.perf_counter() - preprocess_started

    prefill_started = time.perf_counter()

    def prefill() -> None:
        draft_model.eval(prefix_tokens)
        target_model.eval(prefix_tokens)

    _, prefill_energy, prefill_energy_duration = _measure(energy_meter, prefill)
    prefill_time = time.perf_counter() - prefill_started

    output_tokens = list(prefix_tokens)
    generated_tokens: List[int] = []
    accepted_total = 0
    verify_spec_lengths: List[int] = []
    verify_accept_lengths: List[int] = []
    verification_trace: List[Dict[str, Any]] = []
    batch_trace: List[Dict[str, Any]] = []
    token_durations: List[float] = []
    rollback_events = 0
    draft_tokens_generated = 0
    ended_with_eos = False
    first_token_latency = None

    decode_started = time.perf_counter()
    previous_round_finished_at = decode_started

    def decode() -> None:
        nonlocal accepted_total, rollback_events, draft_tokens_generated
        nonlocal ended_with_eos, first_token_latency, previous_round_finished_at
        while (
            accepted_total < accepted_token_budget
            and len(generated_tokens) < max_generated_tokens
        ):
            round_started = previous_round_finished_at
            confirmed_length = len(output_tokens)
            available_output = max_generated_tokens - len(generated_tokens)
            remaining_accepted = accepted_token_budget - accepted_total
            proposal_limit = min(verify_num, available_output, remaining_accepted)
            if proposal_limit <= 0:
                break

            proposal_tokens: List[int] = []
            proposal_probabilities: List[np.ndarray] = []
            for _ in range(proposal_limit):
                probabilities = softmax(
                    draft_model.scores[draft_model.n_tokens - 1]
                )
                token = int(
                    draft_model.sample(
                        top_k=top_k, top_p=top_p, temp=temperature
                    )
                )
                proposal_tokens.append(token)
                proposal_probabilities.append(probabilities)
                draft_model.eval([token])
                draft_tokens_generated += 1
                if token == draft_model.token_eos():
                    break

            verification_started = time.perf_counter()
            result = verify_local_proposal(
                target_model,
                proposal_tokens,
                proposal_probabilities,
                confirmed_length,
                seed=seed,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            verification_seconds = time.perf_counter() - verification_started
            accepted = int(result["n_accepted"])
            final_token = int(result["final_token"])
            accepted_total += accepted
            verify_spec_lengths.append(len(proposal_tokens))
            verify_accept_lengths.append(accepted)
            if accepted < len(proposal_tokens):
                rollback_events += 1

            # Roll the draft model back to the confirmed prefix, retain only the
            # accepted proposal prefix, then consume the target continuation.
            draft_model.n_tokens = confirmed_length + accepted
            draft_model.eval([final_token])

            committed = list(proposal_tokens[:accepted])
            if len(generated_tokens) + len(committed) < max_generated_tokens:
                committed.append(final_token)
            output_tokens.extend(committed)
            generated_tokens.extend(committed)
            previous_round_finished_at = time.perf_counter()
            round_seconds = previous_round_finished_at - round_started
            per_token = round_seconds / len(committed) if committed else round_seconds
            token_durations.extend([per_token] * len(committed))
            if committed and first_token_latency is None:
                first_token_latency = time.perf_counter() - decode_started

            trace_entry = {
                **result,
                "round_id": len(verification_trace),
                "proposal_length": len(proposal_tokens),
                "verification_seconds": verification_seconds,
            }
            verification_trace.append(trace_entry)
            batch_trace.append(
                {
                    "phase": "local_sync_handoff",
                    "actual_batch_size": len(proposal_tokens),
                    "planned_batch_size": verify_num,
                    "should_verify": True,
                    "flush_reason": (
                        "draft_eos"
                        if proposal_tokens[-1] == draft_model.token_eos()
                        else "fixed_window"
                    ),
                }
            )
            if final_token == draft_model.token_eos():
                ended_with_eos = True
                break

    _, decode_energy, decode_energy_duration = _measure(energy_meter, decode)
    decode_time = time.perf_counter() - decode_started

    postprocess_started = time.perf_counter()
    generated_text = draft_model.detokenize(generated_tokens).decode(
        "utf-8", "ignore"
    )
    full_text = draft_model.detokenize(output_tokens).decode("utf-8", "ignore")
    postprocess_time = time.perf_counter() - postprocess_started
    model_energy = (
        float(prefill_energy) + float(decode_energy)
        if prefill_energy is not None and decode_energy is not None
        else None
    )
    energy_duration = (
        float(prefill_energy_duration) + float(decode_energy_duration)
        if prefill_energy_duration is not None and decode_energy_duration is not None
        else None
    )
    return {
        "output_length": len(generated_tokens),
        "accepted_draft_tokens": accepted_total,
        "total_time": decode_time,
        "end_to_end_time_seconds": (
            preprocess_time + prefill_time + decode_time + postprocess_time
        ),
        "request_preprocess_time_seconds": preprocess_time,
        "prefill_time_seconds": prefill_time,
        "decode_time_seconds": decode_time,
        "output_postprocess_time_seconds": postprocess_time,
        "time_to_first_token_seconds": first_token_latency,
        "token_durations": token_durations,
        "output": full_text,
        "generated_text": generated_text,
        "processed_output": generated_text,
        "ended_with_eos": ended_with_eos,
        "generation_cap_hit": (
            len(generated_tokens) >= max_generated_tokens and not ended_with_eos
        ),
        "verify_stats": {"num_verifications": len(verification_trace)},
        "verify_spec_lengths": verify_spec_lengths,
        "verify_accept_lengths": verify_accept_lengths,
        "diagnostics": {
            "rollback_events": rollback_events,
            "draft_tokens_generated": draft_tokens_generated,
            "local_synchronous_handoffs": len(verification_trace),
        },
        "batch_trace": batch_trace,
        "verification_trace": verification_trace,
        "model_energy_joules": model_energy,
        "prompt_prefill_gpu_energy_joules": prefill_energy,
        "nav_gpu_energy_joules": decode_energy,
        "energy_measurement_duration_seconds": energy_duration,
        "energy_source": energy_meter.source,
        "environment_measurements": {},
    }


def build_local_summary(
    mode: str, target_accepted_tokens: int, samples: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    output_tokens = sum(int(item.get("output_length", 0)) for item in samples)
    accepted_tokens = sum(
        sum(int(value) for value in item.get("verify_accept_lengths", []))
        for item in samples
    )
    total_time = sum(float(item.get("total_time", 0.0)) for item in samples)
    verifications = sum(
        int(item.get("verify_stats", {}).get("num_verifications", 0))
        for item in samples
    )
    draft_tokens = sum(sum(item.get("verify_spec_lengths", [])) for item in samples)
    rollbacks = sum(
        int(item.get("diagnostics", {}).get("rollback_events", 0))
        for item in samples
    )
    durations = [
        float(value) for item in samples for value in item.get("token_durations", [])
    ]
    ttft = [
        float(item["time_to_first_token_seconds"])
        for item in samples
        if item.get("time_to_first_token_seconds") is not None
    ]
    batch_sizes = [
        int(batch["actual_batch_size"])
        for item in samples
        for batch in item.get("batch_trace", [])
        if int(batch.get("actual_batch_size", 0)) > 0
    ]
    measured_energy = [
        float(item["model_energy_joules"])
        for item in samples
        if item.get("model_energy_joules") is not None
    ]
    energy = sum(measured_energy) if len(measured_energy) == len(samples) else None
    energy_durations = [
        float(item["energy_measurement_duration_seconds"])
        for item in samples
        if item.get("energy_measurement_duration_seconds") is not None
    ]
    energy_duration = (
        sum(energy_durations) if len(energy_durations) == len(samples) else None
    )
    prefill_energy_values = [
        float(item["prompt_prefill_gpu_energy_joules"])
        for item in samples
        if item.get("prompt_prefill_gpu_energy_joules") is not None
    ]
    prefill_energy = (
        sum(prefill_energy_values)
        if len(prefill_energy_values) == len(samples)
        else None
    )
    decode_energy_values = [
        float(item["nav_gpu_energy_joules"])
        for item in samples
        if item.get("nav_gpu_energy_joules") is not None
    ]
    decode_energy = (
        sum(decode_energy_values)
        if len(decode_energy_values) == len(samples)
        else None
    )
    if mode == "pure_edge":
        energy = None
        energy_duration = None
        prefill_energy = None
        decode_energy = None
    energy_scope = (
        "co_located_cloud_gpu_draft_and_target_prefill_plus_decode"
        if mode == "pure_cloud"
        else "not_measured_no_rapl_permission"
    )
    return {
        "evaluation_protocol": "paper_table1",
        "target_output_tokens": target_accepted_tokens,
        "target_accepted_draft_tokens": target_accepted_tokens,
        "actual_accepted_draft_tokens": accepted_tokens,
        "stopping_criterion": "target_accepted_draft_tokens",
        "actual_output_tokens": output_tokens,
        "num_samples": len(samples),
        "sample_indices": [item.get("sample_index") for item in samples],
        "total_time_seconds": total_time,
        "tpt_normalization_token_type": "target_accepted_draft_tokens",
        "weighted_tpt_seconds": total_time / accepted_tokens if accepted_tokens else None,
        "weighted_tpt_ms": 1000.0 * total_time / accepted_tokens if accepted_tokens else None,
        "accepted_token_tpt_ms": 1000.0 * total_time / accepted_tokens if accepted_tokens else None,
        "output_token_tpt_ms": 1000.0 * total_time / output_tokens if output_tokens else None,
        "throughput_tokens_per_second": accepted_tokens / total_time if total_time else None,
        "accepted_tokens_per_second": accepted_tokens / total_time if total_time else None,
        "output_tokens_per_second": output_tokens / total_time if total_time else None,
        "token_latency_p50_seconds": percentile(durations, 0.50),
        "token_latency_p95_seconds": percentile(durations, 0.95),
        "token_latency_p99_seconds": percentile(durations, 0.99),
        "mean_ttft_seconds": statistics.fmean(ttft) if ttft else None,
        "num_verifications": verifications,
        "verification_frequency": verifications / accepted_tokens if accepted_tokens else None,
        "mean_draft_length": draft_tokens / verifications if verifications else None,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        "rollback_rate": rollbacks / verifications if verifications else None,
        "mean_actual_batch_size": statistics.fmean(batch_sizes) if batch_sizes else None,
        "cap_hit_count": sum(bool(item.get("generation_cap_hit")) for item in samples),
        "cap_hit_rate": (
            sum(bool(item.get("generation_cap_hit")) for item in samples) / len(samples)
            if samples else None
        ),
        "eos_count": sum(bool(item.get("ended_with_eos")) for item in samples),
        "model_energy_joules": energy,
        "gpu_energy_joules": energy if mode == "pure_cloud" else None,
        "prompt_prefill_gpu_energy_joules": prefill_energy,
        "nav_gpu_energy_joules": decode_energy,
        "energy_measurement_duration_seconds": energy_duration,
        "energy_scope": energy_scope,
        "energy_source": (
            "nvml_gpu_board_power"
            if energy is not None
            else ("not_measured" if mode == "pure_edge" else "unavailable")
        ),
        "energy_normalization_token_type": "target_accepted_draft_tokens",
        "energy_included_stages": [
            "local_draft_and_target_prompt_prefill",
            "local_draft_generation_and_target_verification",
        ] if mode == "pure_cloud" else [],
        "energy_excluded_stages": [
            "model_load",
            "prompt_tokenization",
            "output_detokenization",
        ] if mode == "pure_cloud" else ["all_pure_edge_energy"],
        "model_energy_joules_per_100_tokens": (
            100.0 * energy / accepted_tokens
            if energy is not None and accepted_tokens
            else None
        ),
    }


class LocalSpeculativeEvaluator:
    def __init__(self, args, model_factory=None, energy_meter=None):
        self.args = args
        self.mode = args.mode
        self.dataset = args.dataset.lower()
        self.run_id = args.run_id or uuid.uuid4().hex
        self.data_path = Path(args.data_path or f"data/{self.dataset}.jsonl")
        defaults = DEFAULT_MODEL_PAIRS[self.dataset]
        self.draft_path = Path(args.draft_model_path or defaults["draft"])
        self.target_path = Path(args.target_model_path or defaults["target"])
        self.samples = load_samples(
            self.data_path,
            self.dataset,
            args.start_index_of_sample,
            args.end_index_of_sample,
            args.max_samples,
        )
        if not self.samples:
            raise RuntimeError("no dataset samples are available")
        if model_factory is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("llama-cpp-python is required for local SD") from exc
            model_factory = Llama

        cloud = self.mode == "pure_cloud"
        draft_layers = args.draft_n_gpu_layers
        target_layers = args.target_n_gpu_layers
        if draft_layers is None:
            draft_layers = -1 if cloud else 0
        if target_layers is None:
            target_layers = -1 if cloud else 0
        common = {
            "n_threads": args.threads,
            "n_threads_batch": args.threads,
            "verbose": False,
            "logits_all": True,
            "n_ctx": args.ctx_size,
            "seed": args.seed,
        }
        started = time.time()
        self.draft_model = model_factory(
            model_path=str(self.draft_path), n_gpu_layers=draft_layers, **common
        )
        self.draft_load_seconds = time.time() - started
        started = time.time()
        self.target_model = model_factory(
            model_path=str(self.target_path), n_gpu_layers=target_layers, **common
        )
        self.target_load_seconds = time.time() - started
        self.draft_n_gpu_layers = draft_layers
        self.target_n_gpu_layers = target_layers
        self.energy_meter = energy_meter or select_energy_meter(
            self.mode,
            args.gpu_energy_device,
            args.energy_sample_interval,
            target_layers,
        )

    def _manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": 2,
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
            "decoding_algorithm": "fixed_window_speculative_decoding",
            "verification_strategy": "fixed_num_synchronous",
            "verify_num": self.args.verify_num,
            "seed": self.args.seed,
            "evaluation_protocol": "paper_table1",
            "target_accepted_draft_tokens": self.args.target_output_tokens,
            "timing_scope": "decode_after_both_prompt_prefills_excluding_model_load",
            "network_included": False,
            "network_shaping_mode": None,
            "draft_model_path": str(self.draft_path),
            "draft_model_sha256": sha256_file(self.draft_path),
            "draft_n_gpu_layers": self.draft_n_gpu_layers,
            "draft_model_load_seconds": self.draft_load_seconds,
            "target_model_path": str(self.target_path),
            "target_model_sha256": sha256_file(self.target_path),
            "target_n_gpu_layers": self.target_n_gpu_layers,
            "target_model_load_seconds": self.target_load_seconds,
            "data_path": str(self.data_path),
            "data_sha256": sha256_file(self.data_path),
            "result_tag": self.args.result_tag,
            "arguments": vars(self.args),
        }

    def _result_path(self) -> Path:
        run_id = "".join(
            character
            for character in self.run_id
            if character.isalnum() or character in "-_"
        )
        tag = f"_tag={self.args.result_tag}" if self.args.result_tag else ""
        return (
            Path(self.args.output_root)
            / self.dataset
            / self.mode
            / f"{self.mode}{tag}_run={run_id}.json"
        )

    def run(self) -> Tuple[Dict[str, Any], Path]:
        target = int(self.args.target_output_tokens)
        if target <= 0:
            raise ValueError("target_output_tokens must be positive")
        accepted = 0
        no_progress = 0
        results: List[Dict[str, Any]] = []
        for sample in itertools.cycle(self.samples):
            if accepted >= target:
                break
            result = generate_one_local_sd(
                self.draft_model,
                self.target_model,
                sample["prompt"],
                target - accepted,
                verify_num=int(self.args.verify_num),
                max_generated_tokens=int(self.args.max_generated_tokens),
                seed=int(self.args.seed),
                top_k=int(self.args.top_k),
                top_p=float(self.args.top_p),
                temperature=float(self.args.temp),
                energy_meter=self.energy_meter,
            )
            newly_accepted = int(result["accepted_draft_tokens"])
            if newly_accepted > target - accepted:
                raise RuntimeError("local target exceeded the remaining accepted-token budget")
            result.update(
                {
                    "task_id": sample["task_id"],
                    "dataset_task_id": sample["task_id"],
                    "sample_index": sample["sample_index"],
                    "cumulative_accepted_draft_tokens": accepted + newly_accepted,
                }
            )
            for key in ("reference_answer", "canonical_solution", "entry_point"):
                if key in sample:
                    result[key] = sample[key]
            results.append(result)
            if newly_accepted <= 0:
                no_progress += 1
                if no_progress >= len(self.samples):
                    raise RuntimeError("a complete dataset pass produced no accepted tokens")
                continue
            no_progress = 0
            accepted += newly_accepted

        if accepted != target:
            raise RuntimeError(f"accepted {accepted} draft tokens, expected {target}")
        payload = {
            "manifest": self._manifest(),
            "summary": build_local_summary(self.mode, target, results),
            "samples": results,
        }
        path = self._result_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload, path

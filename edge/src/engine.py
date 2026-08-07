import os
import json
import torch
import transformers
import warnings
transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from abc import ABC, abstractmethod
from .util import seed_everything, softmax, strategy2exp
from .merge import OnlineEnvironmentEstimator, PaperDPScheduler, next_plan_index
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import msgpack
import threading
from .comm import BandwidthSender
from .software_link import SoftwareLink
# For GGUF model support
try:
    from llama_cpp import Llama
    GGUF_SUPPORT = True
except ImportError:
    GGUF_SUPPORT = False
    print("Warning: llama-cpp-python not found. GGUF model support disabled.")

# ========== 配置 ==========
def resolve_server_url() -> str:
    return os.getenv("PIPE_SD_SERVER_URL", "http://127.0.0.1:8000")


# URL = "http://39.102.209.27:6001"  # 你的云端 FastAPI 地址
# URL = "http://106.63.100.63:30007"
URL = resolve_server_url()

INIT_ENDPOINT = f"{URL}/init"
START_ENDPOINT = f"{URL}/start"
PROPOSE_ENDPOINT = f"{URL}/propose"
EXIT_ENDPOINT = f"{URL}/exit"
DELAY_ENDPOINT = f"{URL}/delay"

max_probs = []


def _parse_bandwidth_profile(value: str) -> List[Tuple[float, float]]:
    profile: List[Tuple[float, float]] = []
    for raw_pair in str(value or "").split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        try:
            uplink, downlink = (float(item.strip()) for item in raw_pair.split(":", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "software_bandwidth_profile must use up_MBps:down_MBps pairs"
            ) from exc
        if uplink <= 0 or downlink <= 0:
            raise ValueError("software bandwidth profile rates must be positive")
        profile.append((uplink, downlink))
    return profile

def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        
        seed_everything(self.args.seed)
        self.seed = self.args.seed

        self.draft_model = None

        self.gamma: int = getattr(args, 'gamma', 5)
        self.max_generated_len: int = getattr(args, 'max_generated_tokens', 512)
        self.top_k: int = getattr(args, 'top_k', 40)
        self.top_p: float = getattr(args, 'top_p', 0.95)
        self.temp: float = getattr(args, 'temp', 0)
        self.C: float = getattr(args, 'C', 0.05)
        self.verify_strategy: str = getattr(args, 'verify_strategy', "fixed-num")
        self.verify_num: int = getattr(args, 'verify_num', 8)
        self.accumulated_probs: float = 0.0
        self.bandwidth_MBps: float = getattr(args, 'bandwidth_MBps', 2)
        self.downlink_bandwidth_MBps: float = getattr(args, 'downlink_bandwidth_MBps', 25.0)
        self.network_shaping_mode: str = getattr(args, 'network_shaping_mode', 'software')
        configured_uplink_startup_ms = getattr(args, 'software_uplink_startup_ms', None)
        self.software_uplink_startup_seconds = (
            self.C if configured_uplink_startup_ms is None
            else max(0.0, float(configured_uplink_startup_ms) / 1000.0)
        )
        self.software_downlink_startup_seconds = max(
            0.0, float(getattr(args, 'software_downlink_startup_ms', 0.0)) / 1000.0
        )
        self.software_bandwidth_profile = _parse_bandwidth_profile(
            getattr(args, 'software_bandwidth_profile', '')
        )
        if self.software_bandwidth_profile:
            profile_offset = int(
                getattr(args, 'software_bandwidth_profile_offset', 0)
            ) % len(self.software_bandwidth_profile)
            self.software_bandwidth_profile = (
                self.software_bandwidth_profile[profile_offset:]
                + self.software_bandwidth_profile[:profile_offset]
            )
        # Start a dynamic trace only after model loading, so every method sees
        # profile entry zero at the beginning of measured inference.
        self._network_profile_started_at = None
        self.merge_policy: str = getattr(args, 'merge_policy', 'dp')
        self.verify_thresh_single = getattr(args, 'verify_thresh_single', 0.94)
        self.verify_thresh_multi = getattr(args, 'verify_thresh_multi', 0.9)
        
        self.multiply_times: float = getattr(args, 'multiply_times', 0.7)
        self.algorithm = getattr(args, 'algorithm', "vanilla")
        self.result_tag: str = getattr(args, 'result_tag', "")
        self.start_index_of_sample = getattr(args, 'start_index_of_sample', 0)
        self.end_index_of_sample = getattr(args, 'end_index_of_sample', 1)
        self._token_time_ref: float = 0.0
        self._run_token_durations: List[float] = []
        self._sample_token_durations: List[float] = []
        # Backward-compatible alias used by older analysis helpers.
        self._token_durations = self._run_token_durations
        self.process_started_at: float = time.time()
        self.process_model_ready_at: float = self.process_started_at
        self.online_environment_measurement = not getattr(self.args, "disable_online_environment_measurement", False)
        self._environment_lock = threading.Lock()
        self._bootstrap_lock = threading.Lock()
        self.environment_estimator = OnlineEnvironmentEstimator(
            history_size=getattr(self.args, "schedule_history_size", 100),
            min_comm_samples=getattr(self.args, "regression_min_comm_samples", 8),
        )
        self.dp_scheduler = PaperDPScheduler(
            alpha=self.software_uplink_startup_seconds + self.software_downlink_startup_seconds,
            beta=(self.args.token_size_MB / self.bandwidth_MBps) if self.bandwidth_MBps else 0.0,
            gamma=getattr(self.args, "initial_generation_gamma", None)
            or getattr(self.args, "default_token_compute", None)
            or 0.036,
            initial_window=getattr(self.args, "schedule_window", 20),
            history_size=getattr(self.args, "schedule_history_size", 100),
            update_threshold=getattr(self.args, "environment_update_threshold", None),
            gamma_update_threshold=getattr(self.args, "gamma_update_threshold", 0.2),
            communication_update_threshold=getattr(self.args, "communication_update_threshold", 0.2),
        )
        # self.exp_name = strategy2exp(self.verify_strategy)
        self.exp_name = os.path.join(os.getcwd(), 'exp', "exp__wjl", self.args.dataset, self.algorithm)
        print(self.exp_name)
        os.makedirs(self.exp_name, exist_ok=True)
        # Only the two proactive methods generate while a NAV request is in
        # flight.  Vanilla/HSL stop generation and upload their whole draft in
        # the NAV request, matching the baseline definitions in the paper.
        self.send_while_generating = self.algorithm in {"edgeLLM", "pipesd"}
        

    def _reset_state(self):
        # record metrics for report
        self.verify_spec_lengths = []
        self.verify_accept_lengths = []
        self.verify_his = []
        self.acc_ratio = 0.0
        self.num_spec_tokens_sent = 0
        self.num_spec_tokens_generated = 0
        self._spec_token_indices_generated = []
        self._spec_token_indices_sent = set()
        self._token_time_ref = 0.0
        self._sample_token_durations = []
        self._sample_decode_started_at = None
        self._first_accepted_token_latency = None
        self._last_commit_included_final_token = False
        self.batch_trace = []
        self.edge_llm_threshold_trace = []
        self._speculative_round_id = 0

        self.verify_thresh_single = self.args.verify_thresh_single
        self.verify_thresh_multi = self.args.verify_thresh_multi

        self.alpha: float = getattr(self.args, 'init_alpha', 0.01)

        self.color_print("[Edge] 工作进程启动，加载模型...", 2)
        start_time = time.time()
        if not self.draft_model:
            self.draft_model = Llama(
                model_path=self.args.draft_model, n_gpu_layers=self.args.draft_n_gpu_layers, n_threads=self.args.threads,
                verbose=False, logits_all=True, n_ctx=self.args.ctx_size,
            )
        end_time = time.time()
        if self.draft_model and self.process_model_ready_at == self.process_started_at:
            self.process_model_ready_at = end_time
        self.color_print(f"[Edge] 模型加载完成，耗时: {end_time - start_time:.2f} 秒", 5)

        software_mode = self.network_shaping_mode == "software"
        if software_mode and self._network_profile_started_at is None:
            self._network_profile_started_at = time.monotonic()
        self.software_link = (
            SoftwareLink(
                uplink_MBps=self.bandwidth_MBps,
                downlink_MBps=self.downlink_bandwidth_MBps,
                uplink_startup_seconds=self.software_uplink_startup_seconds,
                downlink_startup_seconds=self.software_downlink_startup_seconds,
                bandwidth_profile=self.software_bandwidth_profile,
                profile_interval_seconds=getattr(
                    self.args, "software_bandwidth_change_interval_s", 20.0
                ),
                profile_started_at=self._network_profile_started_at,
                history_size=getattr(self.args, "schedule_history_size", 100),
            )
            if software_mode else None
        )
        sender_kwargs = {
            "bandwidth_MBps": self.bandwidth_MBps,
            "base_latency": self.software_uplink_startup_seconds,
            "timeout": getattr(self.args, "server_timeout_s", 10),
            "use_env_proxy": getattr(self.args, "use_env_proxy", False),
            "software_bandwidth_emulation": software_mode,
            "on_complete": self._on_send_measurement if self.online_environment_measurement else None,
            "link": self.software_link,
            "downlink_bandwidth_MBps": self.downlink_bandwidth_MBps,
            "downlink_base_latency": self.software_downlink_startup_seconds,
            "measurement_history_size": getattr(self.args, "schedule_history_size", 100),
        }
        # Pre-NAV traffic remains ordered on the primary channel. A second
        # channel can upload next-round batches while NAV is still running.
        self.sender = BandwidthSender(**sender_kwargs)
        self.proactive_sender = BandwidthSender(**sender_kwargs)
        if self.online_environment_measurement and self._uses_dp_scheduler():
            self._ensure_communication_bootstrap(force_initial=True)

    def _uses_dp_scheduler(self) -> bool:
        return self.merge_policy == "dp" and ("pipesd" in self.algorithm or "merge" in self.algorithm)

    def _uses_pre_nav_pipeline(self) -> bool:
        """Whether draft batches may be uploaded before NAV is triggered."""
        return self.algorithm == "pipesd" and not getattr(self.args, "nomerge", False)

    def _resolve_algorithm_batch_plan(self) -> List[int]:
        """Return the upload plan owned by the selected algorithm.

        PipeSD alone uses the paper DP scheduler before and during NAV.  The
        EdgeLLM adaptation uses one moving-average N-hat batching window only
        for proactive uploads made while NAV is pending.  Vanilla and HSL do
        not consume this plan because they upload once, at NAV.
        """
        if self._uses_pre_nav_pipeline():
            return self._resolve_merge_plan()
        if self.algorithm == "edgeLLM":
            return [max(1, int(self.dp_scheduler.window))]
        return [max(1, int(self.max_generated_len))]

    def _ensure_communication_bootstrap(self, force_initial: bool = False) -> None:
        """Collect the paper's 1--8 token-batch communication probes."""
        if not self.online_environment_measurement or not self._uses_dp_scheduler() or not hasattr(self, "sender"):
            return
        with self._bootstrap_lock:
            with self._environment_lock:
                sample_count = len(self.environment_estimator.comm_samples)
                missing_sizes = self.environment_estimator.missing_batch_sizes(range(1, 9))
                history_full = sample_count >= self.environment_estimator.history_size
            if force_initial and sample_count > 0:
                force_initial = False
            if not missing_sizes or (not force_initial and not history_full):
                return

            vocab_size = int(getattr(self.args, 'vocab_size', 32000))
            probe_probs = [0.0] * vocab_size
            for batch_size in missing_sizes:
                payload = msgpack.packb({
                    'type': 'communication_probe',
                    'tokens': [0] * batch_size,
                    'probs': [probe_probs] * batch_size,
                    'token_count': batch_size,
                })
                future = self.sender.submit(
                    DELAY_ENDPOINT,
                    payload,
                    headers={"Content-Type": "application/octet-stream"},
                    tag=f"comm-bootstrap-{batch_size}",
                    token_count=batch_size,
                    measurement_kind="transport",
                )
                future.result()

    def _resolve_merge_plan(self) -> List[int]:
        if self.merge_policy == "immediate":
            return [1] * 40
        if self.merge_policy == "no_early":
            return [100]

        self._ensure_communication_bootstrap()
        with self._environment_lock:
            return self.dp_scheduler.plan()

    def _observe_completed_draft_round(self, draft_length: int) -> None:
        if self.merge_policy == "dp":
            with self._environment_lock:
                self.dp_scheduler.observe_draft_length(draft_length)

    def _on_send_measurement(self, measurement: Dict[str, object]) -> None:
        if (
            not self.online_environment_measurement
            or not measurement.get("success")
            or measurement.get("measurement_kind") != "transport"
        ):
            return
        token_count = measurement.get("token_count")
        elapsed_seconds = measurement.get("elapsed_seconds")
        if token_count is None or elapsed_seconds is None:
            return
        with self._environment_lock:
            self.environment_estimator.observe_communication(int(token_count), float(elapsed_seconds))
            estimates = self.environment_estimator.estimate()
            if estimates:
                self.dp_scheduler.update_parameters(**estimates)

    def _observe_generation_time(self, token_count: int, elapsed_seconds: float) -> None:
        if not self.online_environment_measurement:
            return
        with self._environment_lock:
            self.environment_estimator.observe_generation(token_count, elapsed_seconds)
            estimates = self.environment_estimator.estimate()
            if estimates:
                self.dp_scheduler.update_parameters(**estimates)

    def _environment_snapshot(self) -> Dict[str, object]:
        with self._environment_lock:
            return {
                "online_measurement_enabled": self.online_environment_measurement,
                "estimator": self.environment_estimator.snapshot(),
                "dp_scheduler": self.dp_scheduler.snapshot(),
                "software_link": self.software_link.snapshot() if self.software_link else None,
                "primary_sender": self.sender.snapshot() if hasattr(self, "sender") else None,
                "proactive_sender": self.proactive_sender.snapshot() if hasattr(self, "proactive_sender") else None,
            }

    def _network_configuration_snapshot(self) -> Dict[str, object]:
        return {
            "mode": self.network_shaping_mode,
            "emulator_version": SoftwareLink.VERSION if self.network_shaping_mode == "software" else None,
            "queue_policy": "shared-fifo-per-direction" if self.network_shaping_mode == "software" else "os-managed",
            "uplink_bandwidth_MBps": self.bandwidth_MBps,
            "downlink_bandwidth_MBps": self.downlink_bandwidth_MBps,
            "uplink_startup_seconds": self.software_uplink_startup_seconds if self.network_shaping_mode == "software" else None,
            "downlink_startup_seconds": self.software_downlink_startup_seconds if self.network_shaping_mode == "software" else None,
            "bandwidth_profile": [list(item) for item in self.software_bandwidth_profile],
            "bandwidth_change_interval_seconds": (
                float(getattr(self.args, "software_bandwidth_change_interval_s", 20.0))
                if self.software_bandwidth_profile else None
            ),
        }

    def _record_token_time(self, token_count: int) -> None:
        if token_count <= 0:
            return
        now = time.time()
        elapsed = max(0.0, now - self._token_time_ref)
        per_token = elapsed / token_count
        durations = [per_token] * token_count
        self._sample_token_durations.extend(durations)
        self._run_token_durations.extend(durations)
        self._token_durations = self._run_token_durations
        self._token_time_ref = now

    def _apply_compute_emulation(self) -> None:
        """Apply only the explicit Scenario 2/3 artificial delay."""
        if not getattr(self.args, "enable_compute_emulation", False):
            return
        delay = max(0.0, float(getattr(self.args, "emulated_generation_delay", 0.0)))
        if delay:
            time.sleep(delay)

    def _mark_sent(self, indices: List[int]) -> int:
        new_indices = [idx for idx in indices if idx not in self._spec_token_indices_sent]
        self._spec_token_indices_sent.update(new_indices)
        self.num_spec_tokens_sent += len(new_indices)
        return len(new_indices)

    def _trace_batch(self, *, phase: str, batch_size: int, planned_size: int,
                     plan_index: int, window_id: int, batch_id: int,
                     token_start_index: int, should_verify: bool,
                     flush_reason: Optional[str] = None) -> None:
        self.batch_trace.append({
            "phase": phase,
            "speculative_round_id": self._speculative_round_id,
            "window_id": window_id,
            "batch_id": batch_id,
            "plan_index": plan_index,
            "planned_batch_size": planned_size,
            "actual_batch_size": batch_size,
            "token_start_index": token_start_index,
            "should_verify": should_verify,
            "flush_reason": flush_reason,
            "scheduling_window": int(getattr(self.dp_scheduler, 'window', 0)) if self._uses_dp_scheduler() else None,
            "planned_batches": self.dp_scheduler.plan() if self._uses_dp_scheduler() else None,
        })
    def _commit_verified_tokens(
        self,
        output_tokens: List[int],
        speculative_tokens: List[int],
        n_accepted: int,
        final_token: Optional[int],
    ) -> int:
        """Commit a verification result without exceeding the generation budget."""
        remaining = max(0, self.max_len - len(output_tokens))
        accepted_count = min(max(0, int(n_accepted)), len(speculative_tokens), remaining)
        committed = list(speculative_tokens[:accepted_count])
        output_tokens.extend(committed)
        final_token_committed = False
        if len(committed) < remaining and final_token is not None:
            output_tokens.append(final_token)
            committed.append(final_token)
            final_token_committed = True
        self._last_commit_included_final_token = final_token_committed
        if (
            committed
            and getattr(self, '_first_accepted_token_latency', None) is None
            and getattr(self, '_sample_decode_started_at', None) is not None
        ):
            self._first_accepted_token_latency = max(
                0.0, time.perf_counter() - self._sample_decode_started_at
            )
        self._record_token_time(len(committed))
        return len(committed)

    @staticmethod
    def _is_discarded_proactive_response(response) -> bool:
        return (
            isinstance(response, dict)
            and response.get('status') == 'discarded_stale_proactive_batch'
        )

    def _rollback_discarded_waiting_round(self, output_tokens: List[int]) -> int:
        """Discard an unusable proactive round and restore the verified prefix."""
        self.draft_model.reset()
        self.draft_model.eval(output_tokens)
        self._speculative_round_id += 1
        return self.draft_model.n_tokens

    def _must_verify_for_budget(
        self,
        output_tokens: List[int],
        speculative_tokens: List[int],
        waiting_tokens: Optional[List[int]] = None,
    ) -> bool:
        pending_count = len(speculative_tokens) + len(waiting_tokens or [])
        return len(output_tokens) + pending_count >= self.max_len - 1

    @staticmethod
    def _accepted_budget_remaining(
        max_cloud_accepted_tokens: Optional[int],
        accepted_so_far: int,
    ) -> Optional[int]:
        if max_cloud_accepted_tokens is None:
            return None
        return max(0, int(max_cloud_accepted_tokens) - int(accepted_so_far))

    def _must_verify_for_accepted_budget(
        self,
        speculative_tokens: List[int],
        accepted_so_far: int,
        max_cloud_accepted_tokens: Optional[int],
        waiting_tokens: Optional[List[int]] = None,
    ) -> bool:
        remaining = self._accepted_budget_remaining(
            max_cloud_accepted_tokens, accepted_so_far
        )
        if remaining is None:
            return False
        pending_count = len(speculative_tokens) + len(waiting_tokens or [])
        return pending_count >= remaining

    @staticmethod
    def _advance_accepted_budget(
        accepted_so_far: int,
        newly_accepted: int,
        max_cloud_accepted_tokens: Optional[int],
    ) -> int:
        updated = int(accepted_so_far) + max(0, int(newly_accepted))
        if (
            max_cloud_accepted_tokens is not None
            and updated > int(max_cloud_accepted_tokens)
        ):
            raise RuntimeError(
                "cloud accepted-token budget exceeded inside one sample: "
                f"accepted={updated}, budget={int(max_cloud_accepted_tokens)}"
            )
        return updated

    def _build_histogram(self, values: List[int]) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for value in values:
            key = str(value)
            hist[key] = hist.get(key, 0) + 1
        return hist

    def _build_quantiles(self, values: List[int], probs: Tuple[float, ...] = (0.25, 0.5, 0.75)) -> Dict[str, float]:
        if not values:
            return {}
        arr = np.array(values, dtype=np.float32)
        quantiles = np.quantile(arr, probs)
        return {f"p{int(p * 100)}": float(q) for p, q in zip(probs, quantiles)}

    def _build_verify_diagnostics(self, output_length: int) -> Dict[str, object]:
        spec_lengths = list(self.verify_spec_lengths)
        accept_lengths = list(self.verify_accept_lengths)
        rejected_lengths = [spec - accept for spec, accept in zip(spec_lengths, accept_lengths)]
        num_verifications = len(spec_lengths)

        def mean_or_none(values: List[int]):
            if not values:
                return None
            return float(sum(values) / len(values))

        rollback_events = sum(1 for rejected in rejected_lengths if rejected > 0)
        return {
            'draft_lengths': spec_lengths,
            'accepted_lengths': accept_lengths,
            'rejected_lengths': rejected_lengths,
            'draft_length_hist': self._build_histogram(spec_lengths),
            'accepted_length_hist': self._build_histogram(accept_lengths),
            'rejected_length_hist': self._build_histogram(rejected_lengths),
            'draft_length_quantiles': self._build_quantiles(spec_lengths),
            'accepted_length_quantiles': self._build_quantiles(accept_lengths),
            'rejected_length_quantiles': self._build_quantiles(rejected_lengths),
            'mean_verify_spec_len': mean_or_none(spec_lengths),
            'mean_accept_len': mean_or_none(accept_lengths),
            'mean_rejected_len': mean_or_none(rejected_lengths),
            'rollback_events': rollback_events,
            'rollback_rate': float(rollback_events / num_verifications) if num_verifications else 0.0,
            'verification_frequency': float(num_verifications / output_length) if output_length > 0 else None,
            'accepted_per_verification': mean_or_none(accept_lengths),
            'draft_per_verification': mean_or_none(spec_lengths),
        }

    def _resolve_waiting_verify_length(self, waiting_tokens: List[int], waiting_batch_tokens: List[int]) -> int:
        return len(waiting_tokens)
    
    def if_verify(self, probs_draft, verify_mode):
        """
        判断是否需要进行验证
        """
        if verify_mode == 'fixed-num':
            return len(probs_draft) >= self.verify_num
        elif verify_mode == 'single-token':
            # print("[DEBUG] 单令牌验证检查，最大概率:", probs_draft[-1].max().item())
            return bool(probs_draft[-1].max().item() < self.verify_thresh_single)
        elif verify_mode == 'multiple-tokens':
            # 计算每一行的最大值
            row_maxes = [np.max(x) for x in probs_draft]  # axis=1 表示按行计算
            # 计算所有最大值的乘积
            product = np.prod(row_maxes)
            self.accumulated_probs = product
            return bool(product < self.alpha)
        elif verify_mode == 'hybrid':
            row_maxes = [np.max(x) for x in probs_draft]  # axis=1 表示按行计算
            product = np.prod(row_maxes)
            single_flag = bool(row_maxes[-1] < self.verify_thresh_single)
            multi_flag = bool(product < self.verify_thresh_multi)
            # multi_flag = bool(product < self.alpha)
            return (single_flag or multi_flag)
        elif verify_mode == 'diff':
            max_prob = probs_draft[-1].max().item()
            second_max_prob = np.partition(probs_draft[-1], -2)[-2]
            gap = max_prob - second_max_prob
            return bool(gap < self.verify_thresh_diff)
        elif verify_mode == 'entropy':
            entropy_value = self._entropy_topk(probs_draft[-1], k=10)
            return bool(entropy_value > self.entropy_thresh)
        else:
            raise ValueError(f"Unknown verify_mode: {verify_mode}")

    
    def color_print(self, text: str, color_code: int = 3):
        """
        用指定的颜色在控制台打印文本。
        - color_code: 1:红, 2:绿, 3:黄, 4:蓝, 5:紫, 6:青
        """
        colors = {
            1: "\033[91m",  # Red
            2: "\033[92m",  # Green
            3: "\033[93m",  # Yellow
            4: "\033[94m",  # Blue
            5: "\033[95m",  # Purple
            6: "\033[96m",  # Cyan
        }
        END_COLOR = "\033[0m"
        color = colors.get(color_code, "") # Get color, default to no color
        print(f"{color}{text}{END_COLOR}")
    
    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def preprocess(self, input_text):
        pass
    
    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass
    
    def update_thresh(
        self,
        multiply_times,
        n_accepted,
        n_all,
        accumulated_probs=None,
        phase="primary",
    ):
        if n_all <= 0:
            return None
        confidence = self.accumulated_probs if accumulated_probs is None else accumulated_probs
        confidence = min(1.0, max(1e-300, float(confidence)))
        alpha_before = float(self.alpha)
        round_length = max(1, int(n_all))
        accepted_length = min(round_length, max(0, int(n_accepted)))
        scheduling_window = max(1, int(getattr(self.dp_scheduler, 'window', n_all)))
        if self.algorithm == 'edgeLLM':
            # Appendix G.3 and the authors' released implementation update R1
            # against the draft tokens in this NAV round.  Using PipeSD's
            # moving-average scheduling window here causes short, fully
            # accepted rounds to be misclassified as partial rejections and
            # can lock R1 at 1.0 (one-token NAVs).
            if accepted_length == round_length:
                self.alpha *= getattr(self.args, 'edge_llm_full_accept_decay', 0.5)
            else:
                exponent = (round_length - accepted_length) / round_length
                temp = confidence ** exponent
                if temp > 0:
                    self.alpha /= temp
                else:
                    self.alpha /= 0.5
        elif n_all == n_accepted:
            self.alpha *= multiply_times
        else:
            exponent = max(0.0, (scheduling_window - n_accepted) / scheduling_window)
            temp = confidence ** exponent
            if temp > 0:
                self.alpha /= temp
            else:
                self.alpha /= 0.5
        self.alpha = min(1.0, max(1e-9, self.alpha))

        if self.algorithm != 'edgeLLM':
            return None

        trace = {
            'phase': phase,
            'round_id': int(getattr(self, '_speculative_round_id', 0)),
            'alpha_before': alpha_before,
            'alpha_after': float(self.alpha),
            'cumulative_confidence': confidence,
            'draft_length': round_length,
            'accepted_length': accepted_length,
            'proactive_window_before_observation': scheduling_window,
            'fully_accepted': accepted_length == round_length,
        }
        if not hasattr(self, 'edge_llm_threshold_trace'):
            self.edge_llm_threshold_trace = []
        self.edge_llm_threshold_trace.append(trace)
        return trace
    
    def exp2path(self, bandwidth_label: str):
        merge_suffix = ""
        if self.algorithm == "pipesd":
            merge_suffix = f"_merge={self.merge_policy}"
        tag_suffix = f"_tag={self.result_tag}" if self.result_tag else ""

        if 'edgeLLM' in self.exp_name:
            decay = getattr(self.args, 'edge_llm_full_accept_decay', 0.5)
            saved_path = os.path.join(self.exp_name, f"edgeLLM_alpha={self.args.init_alpha}_decay={decay}{tag_suffix}_bw={bandwidth_label}MB.json")
        elif 'single' in self.exp_name or 'hsl' in self.exp_name:
            saved_path = os.path.join(self.exp_name, f"st={self.verify_thresh_single}{tag_suffix}_bw={bandwidth_label}MB.json")
        elif 'hybrid' in self.exp_name or 'pipesd' in self.exp_name:
            if self.args.ablation_study:
                if self.args.verify_strategy == 'single-token':
                    saved_path = os.path.join(self.exp_name, f"ab_single_st={self.verify_thresh_single}{merge_suffix}{tag_suffix}_bw={bandwidth_label}MB.json")
                elif self.args.verify_strategy == 'multiple-tokens':
                    saved_path = os.path.join(self.exp_name, f"ab_multi_ia={self.args.init_alpha}_mt={self.multiply_times}{merge_suffix}{tag_suffix}_bw={bandwidth_label}MB.json")
                elif self.args.verify_strategy == 'fixed-num':
                    saved_path = os.path.join(self.exp_name, f"ab_fixed_gamma={self.gamma}{merge_suffix}{tag_suffix}_bw={bandwidth_label}MB.json")
                else:
                    saved_path = os.path.join(self.exp_name, f"ab_nomerge{merge_suffix}{tag_suffix}_bw={bandwidth_label}MB.json")
            else:
                if self.args.bayes_optimize:
                    saved_path = os.path.join(self.exp_name, f"bc={self.args.bayes_calls}_bound={self.args.bayes_single_min}-{self.args.bayes_single_max}-{self.args.bayes_multi_min}-{self.args.bayes_multi_max}{merge_suffix}{tag_suffix}_bw={bandwidth_label}MB.json")
                else:
                    saved_path = os.path.join(self.exp_name, f"st={self.verify_thresh_single}_mt={self.verify_thresh_multi}{merge_suffix}{tag_suffix}_bw={bandwidth_label}MB.json")
        else:
            saved_path = os.path.join(self.exp_name, f"gamma_{self.gamma}{tag_suffix}_bw={bandwidth_label}MB.json")

        return saved_path
    
    def edge_process_draft_model(
        self,
        prefix,
        task_id,
        persist_result: bool = True,
        max_accepted_tokens: Optional[int] = None,
        max_cloud_accepted_tokens: Optional[int] = None,
    ):
        """
        [最终修正 & 多任务版] 边缘端工作进程。
        一个一个生成token，满足发送条件就发送，满足验证条件时验证。
        """
        
        if prefix is None:
            exit_payload = msgpack.packb({'type': 'exit', 'task_id': task_id})
            exit_future = self.sender.submit(
                EXIT_ENDPOINT,
                exit_payload,
                headers={"Content-Type": "application/msgpack"},
            )
            exit_future.result()
            return

        self.draft_model.reset()
        output_tokens = self.draft_model.tokenize(prefix.encode("utf-8"), add_bos=True)
        prefix_len = len(output_tokens)
        generation_budget = self.max_generated_len
        # max_accepted_tokens is retained as the legacy per-output cap used by
        # BO/debug callers.  The paper protocol uses the independent cloud-
        # accepted budget and leaves this output cap at max_generated_len (128).
        if max_accepted_tokens is not None:
            generation_budget = min(generation_budget, max(0, int(max_accepted_tokens)))
        cloud_accepted_budget = (
            max(0, int(max_cloud_accepted_tokens))
            if max_cloud_accepted_tokens is not None
            else None
        )
        self.max_len = prefix_len + generation_budget
        self.color_print(f"[Edge] 任务 {task_id} 开始处理，prefix 长度 {len(output_tokens)}", 2)

        init_payload = {'type': 'init', 'tokens': output_tokens, 'task_id': task_id}
        init_future = self.sender.submit(
            INIT_ENDPOINT,
            init_payload,
            headers={"Content-Type": "application/json"},
        )
        res = init_future.result()
        if res is None or 'error' in res:
            print("[Edge] 初始化请求失败，退出进程。")
            return
        print(res)

        self.draft_model.eval(output_tokens)
        current_n_past = self.draft_model.n_tokens  # 当前的n_past

        # 全局的推测token序列（用于验证）
        total_speculative_tokens = []  # 用于验证的总推测token
        total_speculative_probs = []   # 对应的概率分布
        total_speculative_indices = []
        
        # 当前批次的数据（用于发送）
        current_batch_tokens = []
        current_batch_probs = []
        current_batch_indices = []

        merge_plan_batches = self._resolve_algorithm_batch_plan()
        if self._uses_pre_nav_pipeline():
            print(f"[Edge] 计算得到合并计划: {merge_plan_batches}")

        merge_plan_index = 0
        merge_window_id = 0
        merge_batch_id = 0

        energy_start_result = self.sender.submit(
            START_ENDPOINT,
            {'task_id': task_id},
            headers={"Content-Type": "application/json"},
        ).result()
        if energy_start_result is None or 'error' in energy_start_result:
            raise RuntimeError(f"cloud task timing start failed: {energy_start_result}")

        # Measure generation only after model loading, communication bootstrap,
        # cloud prompt initialization, and the initial DP plan are ready.
        self._token_time_ref = time.time()
        total_start_time = self._token_time_ref
        self._sample_decode_started_at = time.perf_counter()
        sample_cloud_accepted_tokens = 0
        accepted_budget_is_binding = (
            cloud_accepted_budget is not None
            and cloud_accepted_budget <= generation_budget
        )
        allow_waiting_generation = (
            self.send_while_generating and not accepted_budget_is_binding
        )
        while (
            len(output_tokens) < self.max_len
            and (
                cloud_accepted_budget is None
                or sample_cloud_accepted_tokens < cloud_accepted_budget
            )
        ):

            # --- 1. 生成一个token ---
            generation_step_start = time.perf_counter()
            next_token = self.draft_model.sample(top_k=self.top_k, top_p=self.top_p, temp=self.temp)
            current_probs = softmax(self.draft_model.scores[self.draft_model.n_tokens-1])
            self.num_spec_tokens_generated += 1
            self._spec_token_indices_generated.append(self.num_spec_tokens_generated)
            
            # 添加到全局推测序列
            total_speculative_tokens.append(next_token)
            total_speculative_indices.append(self._spec_token_indices_generated[-1])
            current_batch_tokens.append(next_token)
            current_batch_indices.append(self._spec_token_indices_generated[-1])
            
            self.draft_model.eval([next_token])
            self._observe_generation_time(1, time.perf_counter() - generation_step_start)
            self._apply_compute_emulation()
            
            # 获取概率分布
            
            current_batch_probs.append(current_probs)
            total_speculative_probs.append(current_probs)

            max_probs.append(current_probs.max().item())
            
            if 'edgeLLM' in self.algorithm:
                self.verify_thresh_multi = self.alpha  # edgeLLM中多项式阈值等于alpha

            # --- 2. 检查发送和验证条件 ---
            should_send = (
                self._uses_pre_nav_pipeline()
                and len(current_batch_tokens) >= merge_plan_batches[merge_plan_index]
            )

            should_verify = self.if_verify(
                total_speculative_probs,
                self.verify_strategy
            )
            should_verify = should_verify or self._must_verify_for_budget(
                output_tokens,
                total_speculative_tokens,
            )
            should_verify = should_verify or self._must_verify_for_accepted_budget(
                total_speculative_tokens,
                sample_cloud_accepted_tokens,
                cloud_accepted_budget,
            )

            should_end = (next_token == self.draft_model.token_eos())  # 结束条件

            # 如果同时满足发送和验证条件，优先验证（因为验证需要处理结果）
            if should_verify or should_end:
                # The waiting loop computes confidence for the next round, so
                # retain the C1 belonging to this NAV before it can be replaced.
                nav_accumulated_probs = self.accumulated_probs
                nav_round_id = self._speculative_round_id
                active_plan_index = merge_plan_index
                active_planned_size = merge_plan_batches[active_plan_index]
                # 发起验证请求 - 使用当前的n_past值
                payload = {
                    'type': 'propose',
                    'tokens': current_batch_tokens.copy(),
                    'probs': [p.tolist() for p in current_batch_probs],
                    'task_id': task_id,
                    'n_past': current_n_past,  # 使用当前的n_past
                    'index': len(total_speculative_tokens) - len(current_batch_tokens),  # 本次验证的索引（从0开始）
                    'should_verify': True,  # 验证请求
                    'speculative_round_id': nav_round_id,
                    'window_id': merge_window_id,
                    'batch_id': merge_batch_id,
                    'token_start_index': len(total_speculative_tokens) - len(current_batch_tokens),
                    'token_count': len(current_batch_tokens),
                    'prefix_version': current_n_past,
                }
                payload_bytes = msgpack.packb(payload)
                # 将本轮所有推测token的索引计入“已验证”集合（避免重复计数）
                self._mark_sent(total_speculative_indices)
                self._trace_batch(
                    phase='draft',
                    batch_size=len(current_batch_tokens),
                    planned_size=active_planned_size,
                    plan_index=active_plan_index,
                    window_id=merge_window_id,
                    batch_id=merge_batch_id,
                    token_start_index=payload['index'],
                    should_verify=True,
                    flush_reason='eos' if should_end else 'nav',
                )
                merge_plan_index = 0
                merge_window_id += 1
                merge_batch_id = 0

                # print(f"[DEBUG] 发送验证请求，tokens: {current_batch_tokens}, n_past: {current_n_past}， tokens: {self.draft_model.detokenize(total_speculative_tokens).decode('utf-8', 'ignore')}")

                future = self.sender.submit(
                    PROPOSE_ENDPOINT,
                    payload_bytes,
                    headers={"Content-Type": "application/msgpack"},
                    token_count=len(current_batch_tokens),
                    measurement_kind="nav",
                )
                verify_result = None
                
                # --- 3. 在等待验证结果期间继续生成token ---
                waiting_tokens = []  # 等待期间生成的token
                waiting_probs = []   # 等待期间的token概率
                waiting_indices = []
                speculated_final_token = None
                waiting_tag = f"wait_{task_id}_{current_n_past}"
                waiting_futures = []
                waiting_batch_tokens = None
                waiting_batch_probs = None
                waiting_batch_indices = None
                should_verify_waiting = False
                waiting_verify_future = None
                waiting_plan_batches = list(merge_plan_batches)
                waiting_plan_index = 0
                waiting_window_id = 0
                waiting_batch_id = 0
                waiting_pending_tokens = []
                waiting_pending_probs = []
                waiting_pending_indices = []
                waiting_accumulated_probs = None
                nav_returned = False
                self._speculative_round_id += 1

                while (
                    allow_waiting_generation
                    and len(output_tokens)
                    + len(total_speculative_tokens)
                    + len(waiting_tokens)
                    + (2 if not waiting_tokens else 1)
                    <= self.max_len
                ):

                    if not waiting_tokens:
                        speculative_final_step_start = time.perf_counter()
                        speculated_final_token = self.draft_model.sample(top_k=self.top_k, top_p=self.top_p, temp=self.temp)
                        self.draft_model.eval([speculated_final_token])
                        self._observe_generation_time(
                            1,
                            time.perf_counter() - speculative_final_step_start,
                        )
                        self._apply_compute_emulation()
                        self.num_spec_tokens_generated += 1
                        self._spec_token_indices_generated.append(self.num_spec_tokens_generated)
                    
                    waiting_step_start = time.perf_counter()
                    wait_token = self.draft_model.sample(top_k=self.top_k, top_p=self.top_p, temp=self.temp)
                    wait_probs = softmax(self.draft_model.scores[self.draft_model.n_tokens-1])
                    self.num_spec_tokens_generated += 1
                    self._spec_token_indices_generated.append(self.num_spec_tokens_generated)
                    
                    # eval这个等待token
                    self.draft_model.eval([wait_token])
                    self._observe_generation_time(1, time.perf_counter() - waiting_step_start)
                    self._apply_compute_emulation()
                    
                    waiting_tokens.append(wait_token)
                    waiting_probs.append(wait_probs)
                    waiting_indices.append(self._spec_token_indices_generated[-1])
                    waiting_pending_tokens.append(wait_token)
                    waiting_pending_probs.append(wait_probs)
                    waiting_pending_indices.append(self._spec_token_indices_generated[-1])
                    
                    if 'edgeLLM' in self.algorithm:
                        self.verify_thresh_multi = self.alpha  # edgeLLM中多项式阈值等于alpha
                    
                    should_verify_waiting = self.if_verify(
                        waiting_probs,
                        self.verify_strategy
                    )
                    waiting_accumulated_probs = self.accumulated_probs
                    should_verify_waiting = should_verify_waiting or self._must_verify_for_budget(
                        output_tokens,
                        total_speculative_tokens,
                        waiting_tokens,
                    )
                    should_verify_waiting = (
                        should_verify_waiting
                        or self._must_verify_for_accepted_budget(
                            total_speculative_tokens,
                            sample_cloud_accepted_tokens,
                            cloud_accepted_budget,
                            waiting_tokens,
                        )
                    )
                    should_verify_waiting = should_verify_waiting or wait_token == self.draft_model.token_eos()

                    if future.done():
                        verify_result = future.result()
                        if 'error' in verify_result or 'n_accepted' not in verify_result or 'final_token' not in verify_result:
                            print(f"[Edge] 服务器返回错误: {verify_result}")
                            return
                        n_accepted = verify_result['n_accepted']
                        final_token = verify_result['final_token']
                        if not (n_accepted == len(total_speculative_tokens) and final_token == speculated_final_token):
                            break
                        nav_returned = True

                    planned_waiting_size = waiting_plan_batches[waiting_plan_index]
                    should_flush_waiting = (
                        len(waiting_pending_tokens) >= planned_waiting_size
                        or should_verify_waiting
                        or nav_returned
                    )
                    if should_flush_waiting:
                        waiting_batch_tokens = list(waiting_pending_tokens)
                        waiting_batch_probs = list(waiting_pending_probs)
                        waiting_batch_indices = list(waiting_pending_indices)
                        waiting_start_index = len(waiting_tokens) - len(waiting_batch_tokens)
                        waiting_payload = {
                            'type': 'propose_waiting',
                            'tokens': waiting_batch_tokens,
                            'probs': [p.tolist() for p in waiting_batch_probs],
                            'task_id': task_id,
                            'n_past': current_n_past + len(total_speculative_tokens) + 1,
                            'index': waiting_start_index,
                            'should_verify': should_verify_waiting,
                            'speculative_round_id': self._speculative_round_id,
                            'window_id': waiting_window_id,
                            'batch_id': waiting_batch_id,
                            'token_start_index': waiting_start_index,
                            'token_count': len(waiting_batch_tokens),
                            'prefix_version': current_n_past,
                            'expected_prefix_token': speculated_final_token,
                            'parent_round_id': nav_round_id,
                        }
                        waiting_payload_bytes = msgpack.packb(waiting_payload)
                        self._mark_sent(waiting_batch_indices)
                        flush_reason = None
                        if should_verify_waiting:
                            flush_reason = 'waiting_nav'
                        elif nav_returned:
                            flush_reason = 'nav_returned'
                        self._trace_batch(
                            phase='waiting_nav',
                            batch_size=len(waiting_batch_tokens),
                            planned_size=planned_waiting_size,
                            plan_index=waiting_plan_index,
                            window_id=waiting_window_id,
                            batch_id=waiting_batch_id,
                            token_start_index=waiting_start_index,
                            should_verify=should_verify_waiting,
                            flush_reason=flush_reason,
                        )
                        waiting_future = self.proactive_sender.submit(
                            PROPOSE_ENDPOINT,
                            waiting_payload_bytes,
                            headers={"Content-Type": "application/msgpack"},
                            tag=waiting_tag,
                            token_count=len(waiting_batch_tokens),
                            measurement_kind="nav" if should_verify_waiting else "transport",
                        )
                        waiting_futures.append(waiting_future)
                        if should_verify_waiting:
                            waiting_verify_future = waiting_future
                        waiting_pending_tokens = []
                        waiting_pending_probs = []
                        waiting_pending_indices = []
                        previous_waiting_plan_index = waiting_plan_index
                        waiting_plan_index = next_plan_index(waiting_plan_index, waiting_plan_batches)
                        waiting_batch_id += 1
                        if waiting_plan_index == 0 and previous_waiting_plan_index == len(waiting_plan_batches) - 1:
                            waiting_window_id += 1
                            waiting_batch_id = 0

                    if should_verify_waiting or nav_returned:
                        break
                
                if verify_result is None:
                    verify_result = future.result()
                
                # 检查响应是否包含错误信息
                if 'error' in verify_result or 'n_accepted' not in verify_result:
                    print(f"[Edge] 服务器返回错误: {verify_result}")
                    return

                # 处理验证结果
                n_accepted = verify_result['n_accepted']
                final_token = verify_result['final_token']
                sample_cloud_accepted_tokens = self._advance_accepted_budget(
                    sample_cloud_accepted_tokens,
                    n_accepted,
                    cloud_accepted_budget,
                )
                self.verify_spec_lengths.append(len(total_speculative_tokens))
                self.verify_accept_lengths.append(n_accepted)
                self.verify_his.append((len(total_speculative_tokens), n_accepted))
                self.acc_ratio += n_accepted / len(total_speculative_tokens)
                
                # 更新输出tokens
                self._commit_verified_tokens(
                    output_tokens,
                    total_speculative_tokens,
                    n_accepted,
                    final_token,
                )
                final_token_committed = bool(
                    getattr(self, '_last_commit_included_final_token', False)
                )
                
                last_verify_all_passed = (n_accepted == len(total_speculative_tokens) and final_token == speculated_final_token)
                if not last_verify_all_passed:
                    self.draft_model.n_tokens = current_n_past + n_accepted
                # print(f'当前n_tokens: {self.draft_model.n_tokens}, current_n_past: {current_n_past}, n_accepted: {n_accepted}')
                current_n_past = (
                    current_n_past
                    + n_accepted
                    + (1 if final_token_committed else 0)
                )
                if final_token_committed and final_token != speculated_final_token:
                    self.draft_model.eval([final_token])
                    # print(f"[DEBUG] final_token 与 speculated_final_token 不同，eval final_token: {final_token}")
                
                # print(f"[DEBUG] 更新状态: n_past {current_n_past - n_accepted - 1} -> {current_n_past}, accepted {n_accepted}, final_token {final_token}")
                
                if self.verify_strategy == 'multiple-tokens':
                    # 更新多项式阈值
                    self.update_thresh(
                        multiply_times=self.multiply_times,
                        n_accepted=n_accepted,
                        n_all=len(total_speculative_tokens),
                        accumulated_probs=nav_accumulated_probs,
                        phase="primary",
                    )
                    # print(f"[DEBUG] 更新多项式阈值: verify_thresh_multi={self.verify_thresh_multi:.6f}, accumulated_probs={self.accumulated_probs:.6f}")

                self._observe_completed_draft_round(len(total_speculative_tokens))
                merge_plan_batches = self._resolve_algorithm_batch_plan()
                
                verify_result = None  # 重置验证结果
                
                
                # --- 5. 处理等待期间生成的token ---
                accepted_budget_exhausted = (
                    cloud_accepted_budget is not None
                    and sample_cloud_accepted_tokens >= cloud_accepted_budget
                )
                if len(output_tokens) >= self.max_len or accepted_budget_exhausted:
                    for fut in waiting_futures:
                        if not fut.done():
                            self.proactive_sender.cancel_future(fut)
                    self.proactive_sender.drain_tag(waiting_tag)
                    waiting_tokens = []
                if waiting_tokens:
                    if last_verify_all_passed:
                        if should_verify_waiting:
                            if waiting_verify_future is None:
                                raise RuntimeError("waiting NAV was requested without a flushed verification batch")
                            verify_result_waiting = waiting_verify_future.result()
                            if 'n_accepted' not in verify_result_waiting:
                                if self._is_discarded_proactive_response(verify_result_waiting):
                                    self.color_print(
                                        "[Edge] waiting NAV 已失效，丢弃等待期草稿并恢复到父 NAV 的已验证前缀。",
                                        3,
                                    )
                                    current_n_past = self._rollback_discarded_waiting_round(output_tokens)
                                    waiting_tokens = []
                                else:
                                    raise RuntimeError(
                                        f"invalid waiting NAV response: {verify_result_waiting}"
                                    )
                            else:
                                n_accepted_waiting = verify_result_waiting['n_accepted']
                                final_token_waiting = verify_result_waiting['final_token']
                                sample_cloud_accepted_tokens = self._advance_accepted_budget(
                                    sample_cloud_accepted_tokens,
                                    n_accepted_waiting,
                                    cloud_accepted_budget,
                                )
                                waiting_spec_len = len(waiting_tokens)
                                self.verify_spec_lengths.append(waiting_spec_len)
                                self.verify_accept_lengths.append(n_accepted_waiting)
                                self.verify_his.append((waiting_spec_len, n_accepted_waiting))
                                if waiting_spec_len > 0:
                                    self.acc_ratio += n_accepted_waiting / waiting_spec_len
                                self._commit_verified_tokens(
                                    output_tokens,
                                    waiting_tokens,
                                    n_accepted_waiting,
                                    final_token_waiting,
                                )
                                self.draft_model.reset()
                                self.draft_model.eval(output_tokens)
                                current_n_past = self.draft_model.n_tokens
                                if self.verify_strategy == 'multiple-tokens':
                                    self.update_thresh(
                                        multiply_times=self.multiply_times,
                                        n_accepted=n_accepted_waiting,
                                        n_all=waiting_spec_len,
                                        accumulated_probs=waiting_accumulated_probs,
                                        phase="waiting",
                                    )
                                self._observe_completed_draft_round(waiting_spec_len)
                                merge_plan_batches = self._resolve_algorithm_batch_plan()
                                self._speculative_round_id += 1
                        else:
                            # The cloud has buffered exactly these proactive batches.  Keep
                            # the edge sequence and continue the same speculative round.
                            # Join only at the round boundary: generation and upload were
                            # concurrent throughout NAV, but later primary-channel batches
                            # must not overtake an earlier proactive upload.
                            proactive_round_valid = True
                            for proactive_future in waiting_futures:
                                proactive_result = proactive_future.result()
                                if isinstance(proactive_result, dict) and 'error' in proactive_result:
                                    raise RuntimeError(
                                        f"proactive upload failed: {proactive_result}"
                                    )
                                if self._is_discarded_proactive_response(proactive_result):
                                    proactive_round_valid = False
                            if proactive_round_valid:
                                total_speculative_tokens = list(waiting_tokens)
                                total_speculative_probs = list(waiting_probs)
                                total_speculative_indices = list(waiting_indices)
                                current_batch_tokens = []
                                current_batch_probs = []
                                current_batch_indices = []
                                merge_plan_index = 0 if nav_returned else waiting_plan_index
                                merge_window_id = waiting_window_id
                                merge_batch_id = waiting_batch_id
                            else:
                                self.color_print(
                                    "[Edge] proactive batch 已失效，回退到父 NAV 的已验证前缀。",
                                    3,
                                )
                                current_n_past = self._rollback_discarded_waiting_round(output_tokens)
                                waiting_tokens = []
                    else:
                        for fut in waiting_futures:
                            if not fut.done():
                                self.proactive_sender.cancel_future(fut)
                        self.proactive_sender.drain_tag(waiting_tag)

                if not (last_verify_all_passed and not should_verify_waiting and waiting_tokens):
                    total_speculative_tokens = []
                    total_speculative_probs = []
                    total_speculative_indices = []
                    current_batch_tokens = []
                    current_batch_probs = []
                    current_batch_indices = []


                # 检查是否结束
                if final_token == self.draft_model.token_eos():
                    break

                continue
                    
            elif should_send:
                payload = {
                    'type': 'propose',
                    'tokens': current_batch_tokens.copy(),
                    'probs': [p.tolist() for p in current_batch_probs],
                    'task_id': task_id,
                    'n_past': current_n_past,  # 使用当前的n_past
                    'index': len(total_speculative_tokens) - len(current_batch_tokens),  # 当前批次的索引
                    'should_verify': False,  # 非验证请求
                    'speculative_round_id': self._speculative_round_id,
                    'window_id': merge_window_id,
                    'batch_id': merge_batch_id,
                    'token_start_index': len(total_speculative_tokens) - len(current_batch_tokens),
                    'token_count': len(current_batch_tokens),
                    'prefix_version': current_n_past,
                }
                payload_bytes = msgpack.packb(payload)

                # print(f"[DEBUG] 发送批次请求，tokens: {current_batch_tokens}, n_past: {current_n_past}, index: {len(total_speculative_tokens)}， tokens: {self.draft_model.detokenize(total_speculative_tokens).decode('utf-8', 'ignore')}")
                # print(f"[发送] 当前n_tokens: {self.draft_model.n_tokens}, current_n_past: {current_n_past}")

                future = self.sender.submit(
                    PROPOSE_ENDPOINT,
                    payload_bytes,
                    headers={"Content-Type": "application/msgpack"},
                    token_count=len(current_batch_tokens),
                    measurement_kind="transport",
                )
                self._mark_sent(current_batch_indices)
                self._trace_batch(
                    phase='draft',
                    batch_size=len(current_batch_tokens),
                    planned_size=merge_plan_batches[merge_plan_index],
                    plan_index=merge_plan_index,
                    window_id=merge_window_id,
                    batch_id=merge_batch_id,
                    token_start_index=payload['index'],
                    should_verify=False,
                )

                # 重置当前批次
                current_batch_tokens = []
                current_batch_probs = []
                current_batch_indices = []
                previous_plan_index = merge_plan_index
                merge_plan_index = next_plan_index(merge_plan_index, merge_plan_batches)
                merge_batch_id += 1
                if merge_plan_index == 0 and previous_plan_index == len(merge_plan_batches) - 1:
                    merge_window_id += 1
                    merge_batch_id = 0

        total_end_time = time.time()
        spent_time = total_end_time - total_start_time
        completed_output_length = len(output_tokens) - prefix_len
        tpt = spent_time / completed_output_length if completed_output_length else float('inf')
        self.color_print(f"[Edge] 任务 {task_id} 处理完成，输出长度 {completed_output_length}，总耗时: {spent_time:.4f} 秒, 单位token耗时 {tpt:.4f} 秒", 5)

        decoded_text = self.draft_model.detokenize(output_tokens).decode("utf-8", "ignore")
        post_result = self.postprocess(prefix, decoded_text)
        # self.color_print(f"[Edge] 任务 {task_id} 生成完成, 前缀长度{prefix_len}，输出长度 {len(output_tokens) - prefix_len}，结果:\n{post_result}", 2)
        verify_stats = {
            'num_verifications': len(self.verify_spec_lengths),
            'num_spec_tokens_sent': self.num_spec_tokens_sent,
            'num_spec_tokens_generated': self.num_spec_tokens_generated,
            'num_spec_tokens': self.num_spec_tokens_sent,
            'unique_generated_token_indices': len(set(self._spec_token_indices_generated)),
            'unique_sent_token_indices': len(self._spec_token_indices_sent),
        }
        if self.num_spec_tokens_sent > self.num_spec_tokens_generated:
            raise RuntimeError("sent speculative-token count exceeds generated-token count")
        if sample_cloud_accepted_tokens != sum(self.verify_accept_lengths):
            raise RuntimeError(
                "per-sample cloud accepted-token accounting mismatch: "
                f"tracked={sample_cloud_accepted_tokens}, "
                f"trace={sum(self.verify_accept_lengths)}"
            )

        # Drain proactive uploads before deleting cloud task state so exit
        # metrics include every reused/discarded batch.
        self.proactive_sender.close()
        exit_payload = msgpack.packb({'type': 'exit', 'task_id': task_id})
        exit_result = self.sender.submit(
            EXIT_ENDPOINT,
            exit_payload,
            headers={"Content-Type": "application/msgpack"},
        ).result()

        bandwidth_label = f"{self.bandwidth_MBps:g}"
        # bandwidth_label = "0"
        saved_path = self.exp2path(bandwidth_label)        

        os.makedirs(self.exp_name, exist_ok=True) 
        output_length = len(output_tokens) - prefix_len
        avg_token_time = spent_time / output_length if output_length else None
        diagnostics = self._build_verify_diagnostics(output_length)
        exp_result = {
            'task_id': task_id,
            'client_id': int(getattr(self.args, 'client_id', 0)),
            'dataset_sample_index': getattr(
                self.args, 'current_dataset_sample_index', None
            ),
            'workload_iteration': getattr(
                self.args, 'current_workload_iteration', None
            ),
            'measurement_phase': getattr(
                self.args, 'current_measurement_phase', 'single_pass'
            ),
            'measurement_window_start': getattr(
                self.args, 'measurement_window_start', None
            ),
            'measurement_window_end': getattr(
                self.args, 'measurement_window_end', None
            ),
            'output_length': output_length,
            # 'counted_length': eff_num,
            'total_time': spent_time,
            'sample_started_at': total_start_time,
            'sample_finished_at': total_end_time,
            'process_started_at': self.process_started_at,
            'process_model_ready_at': self.process_model_ready_at,
            'output': decoded_text,
            'generated_text': decoded_text[len(prefix):] if decoded_text.startswith(prefix) else decoded_text,
            'processed_output': post_result,
            'gamma': self.gamma,
            'max_len': self.max_len,
            'requested_output_tokens': generation_budget,
            'output_token_budget': generation_budget,
            'cloud_accepted_token_budget': cloud_accepted_budget,
            'cloud_accepted_tokens': sample_cloud_accepted_tokens,
            'cloud_accepted_budget_reached': bool(
                cloud_accepted_budget is not None
                and sample_cloud_accepted_tokens >= cloud_accepted_budget
            ),
            'output_token_cap_reached': bool(
                completed_output_length >= generation_budget
            ),
            'ended_with_eos': bool(output_tokens and output_tokens[-1] == self.draft_model.token_eos()),
            'generation_cap_hit': bool(
                generation_budget == self.max_generated_len
                and completed_output_length == self.max_generated_len
                and output_tokens
                and output_tokens[-1] != self.draft_model.token_eos()
            ),
            'strategy': self.verify_strategy,
            'merge_policy': self.merge_policy,
            'bandwidth_MBps': self.bandwidth_MBps,
            'thresh_single': self.verify_thresh_single,
            'thresh_multi': self.verify_thresh_multi,
            'verify_stats': verify_stats,
            'token_durations': list(self._sample_token_durations),
            'time_to_first_token_seconds': self._first_accepted_token_latency,
            'avg_token_time': avg_token_time,
            'gpu_power_integral_joules': exit_result.get('gpu_power_integral_joules', None),
            'model_energy_joules': exit_result.get('model_energy_joules', None),
            'prompt_prefill_gpu_energy_joules': exit_result.get(
                'prompt_prefill_gpu_energy_joules'
            ),
            'nav_gpu_energy_joules': exit_result.get('nav_gpu_energy_joules'),
            'prompt_prefill_energy_measurement': exit_result.get(
                'prompt_prefill_energy_measurement'
            ),
            'nav_energy_trace': exit_result.get('nav_energy_trace', []),
            'cloud_batch_trace': exit_result.get('cloud_batch_trace', []),
            'cloud_batch_scheduler': exit_result.get('cloud_batch_scheduler'),
            'energy_measurement_duration_seconds': exit_result.get(
                'energy_measurement_duration_seconds'
            ),
            'energy_measurement_available': exit_result.get(
                'energy_measurement_available', False
            ),
            'energy_scope': exit_result.get(
                'energy_scope', 'cloud_gpu_prompt_prefill_plus_nav_compute'
            ),
            'energy_source': exit_result.get('energy_source', 'nvml_gpu_board_power'),
            'energy_sample_interval_seconds': exit_result.get(
                'energy_sample_interval_seconds'
            ),
            'energy_included_stages': exit_result.get('energy_included_stages', []),
            'energy_excluded_stages': exit_result.get('energy_excluded_stages', []),
            'verify_num': exit_result.get('verify_num', None),
            'cloud_cache_version': exit_result.get('cache_version'),
            'discarded_proactive_tokens': exit_result.get('discarded_proactive_tokens', 0),
            'reused_proactive_tokens': exit_result.get('reused_proactive_tokens', 0),
            'acc_ratio': self.acc_ratio / len(self.verify_spec_lengths) if self.verify_spec_lengths else 0.0,
            'verify_spec_lengths': self.verify_spec_lengths,
            'verify_accept_lengths': self.verify_accept_lengths,
            'verify_his': self.verify_his,
            'edge_llm_threshold_trace': self.edge_llm_threshold_trace,
            'diagnostics': diagnostics,
            'environment_measurements': self._environment_snapshot(),
            'batch_trace': self.batch_trace,
            'compute_emulation': {
                'enabled': bool(getattr(self.args, 'enable_compute_emulation', False)),
                'extra_delay_seconds': float(getattr(self.args, 'emulated_generation_delay', 0.0)),
                'initial_generation_gamma': float(getattr(self.dp_scheduler, 'gamma', 0.0)),
            },
        }
        if not persist_result:
            self.sender.close()
            return _json_safe(exp_result)

        # 读取已有数据
        if os.path.exists(saved_path):
            with open(saved_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # 添加新结果
        data.append(_json_safe(exp_result))

        # 写回整个文件
        with open(saved_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        self.sender.close()

        return post_result, spent_time

        

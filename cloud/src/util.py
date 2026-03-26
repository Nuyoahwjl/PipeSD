import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
import logging
from typing import List, Optional, Tuple

try:
    import pynvml  # type: ignore
    NVML_IMPORTED = True
except ImportError:
    pynvml = None
    NVML_IMPORTED = False

def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # For reproducibility prefer deterministic algorithms and disable
    # the cuDNN auto-tuner which can introduce variability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # If available, enable PyTorch's deterministic mode (may raise on unsupported ops)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions may not have this API; ignore in that case
        pass

def model_zoo(args):
    vocab_size = {
        "TinyLlama-1.1B-Chat-v1.0-GPTQ": 32000,
        "llama-2-7b": 32000,
        "llama-2-70b": 32000,
        "deepseek-1.3b": 32256,
        "deepseek-coder-1.3b-base-GGUF": 32256,
        "deepseek-coder-1.3b-instruct-GGUF": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-coder-6.7B-instruct-GGUF": 32256,
        "deepseek-33b": 32256,
    }
    
    zoo = {
        "llama-2-7b": "pre_models/Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
        "deepseek-6.7b": "pre_models/deepseek-coder-6.7B-instruct-GPTQ",
        "deepseek-coder-6.7B-instruct-GGUF": "pre_models/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
    }
    args.target_model = zoo[args.target_model]

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--dataset', type=str, default="humaneval")
    # parser.add_argument('--dataset', type=str, default="gsm8k")

    parser.add_argument('--draft_model', type=str, default="deepseek-coder-1.3b-instruct-GGUF")
    parser.add_argument('--target_model', type=str, default="llama-2-7b")
    
    parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=40, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=1, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=1, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=4, help='guess time.')
    parser.add_argument("--num_drafts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ctx_size", type=int, default=1024)
    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    args = args_proc(args)
    model_zoo(args)
    return args

def args_proc(args):
    """Process args after parsing."""
    args.data_path = f"data/{args.dataset}.jsonl"

    # Process dataset-specific default models
    if args.dataset == "humaneval":
        args.draft_model = "deepseek-coder-1.3b-instruct-GGUF"
        args.target_model = "deepseek-coder-6.7B-instruct-GGUF"
    elif args.dataset == "mt_bench" or args.dataset == "gsm8k":
        args.draft_model = "tinyllama-1.1b-chat-v1.0-gguf"
        args.target_model = "llama-2-7b"
    return args

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # 防止溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def max_fn(x):
    clipped = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    total = clipped.sum()
    if total <= 0:
        return np.full_like(clipped, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total


def sample(probs, *_args, seed=None, **_kwargs):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        raise ValueError("sample() requires a non-empty probability vector")

    total = probs.sum()
    if total <= 0:
        probs = np.full(probs.shape[0], 1.0 / probs.shape[0], dtype=np.float64)
    else:
        probs = probs / total

    rng = np.random.default_rng(seed)
    return int(rng.choice(probs.shape[0], p=probs))


class GPUEnergyMonitor:
    """Wraps NVML power access so we can compute inference power integrals."""

    def __init__(self, device_index: int = 0, logger: Optional[logging.Logger] = None):
        self.device_index = device_index
        self.enabled = False
        self.handle = None
        self.supports_power = False
        self.logger = logger or logging.getLogger(__name__)

        if not NVML_IMPORTED:
            self.logger.info("pynvml not available; GPU power tracking disabled.")
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            try:
                pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.supports_power = True
            except Exception:
                self.logger.info("GPU power usage sampling not supported on this device.")
            self.enabled = self.supports_power
            if self.enabled:
                self.logger.info("GPU power tracking enabled on device %d", device_index)
        except Exception as exc:
            self.logger.warning("Failed to initialize GPU power tracking: %s", exc)
            self.handle = None
            self.enabled = False

    def read_power_watts(self) -> Optional[float]:
        if not self.enabled or self.handle is None or not self.supports_power:
            return None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0
        except Exception as exc:
            self.logger.warning("Failed to read GPU power usage: %s", exc)
            return None


class EnergyTracker:
    """Context helper that records power integrals for a single inference segment."""

    def __init__(self, monitor: "GPUEnergyMonitor", task: "InferenceTask", stage: str, sample_interval: float, logger: Optional[logging.Logger] = None):
        self.monitor = monitor
        self.task = task
        self.stage = stage
        self.sample_interval = sample_interval
        self._samples: List[Tuple[float, float]] = []
        self._stop_event: Optional[threading.Event] = None
        self._sampler_thread: Optional[threading.Thread] = None
        self.logger = logger or logging.getLogger(__name__)

    def __enter__(self):
        if self.monitor.supports_power:
            self._stop_event = threading.Event()
            self._samples = []

            def sampler():
                while not self._stop_event.is_set():
                    timestamp = time.perf_counter()
                    power = self.monitor.read_power_watts()
                    if power is not None:
                        self._samples.append((timestamp, power))
                    if self._stop_event.wait(self.sample_interval):
                        break

            initial_power = self.monitor.read_power_watts()
            if initial_power is not None:
                self._samples.append((time.perf_counter(), initial_power))
            self._sampler_thread = threading.Thread(target=sampler, daemon=True)
            self._sampler_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stop_event is not None:
            final_power = self.monitor.read_power_watts()
            if final_power is not None:
                self._samples.append((time.perf_counter(), final_power))
            self._stop_event.set()
            if self._sampler_thread is not None:
                self._sampler_thread.join()

        power_integral = 0.0
        if len(self._samples) >= 2:
            for (t0, p0), (t1, p1) in zip(self._samples, self._samples[1:]):
                dt = t1 - t0
                if dt <= 0:
                    continue
                power_integral += (p0 + p1) * 0.5 * dt
            self.task.total_gpu_power_integral_joules += power_integral

        self.logger.info(
            "task=%s stage=%s gpu_power_integral=%.6fJ total_power_int=%.6fJ",
            self.task.task_id,
            self.stage,
            power_integral,
            self.task.total_gpu_power_integral_joules,
        )
        if self.stage == "verify_total":
            self.task.last_verify_power_integral = power_integral

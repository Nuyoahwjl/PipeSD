# communication_service.py
from fastapi import FastAPI, HTTPException, Request
from typing import List, Dict, Optional
from pydantic import BaseModel
import time
import msgpack
import random
import numpy as np
import json
import os
import logging
import threading
import uuid
import torch
from contextlib import nullcontext
from src.util import seed_everything, parse_arguments, softmax, max_fn, sample, GPUEnergyMonitor, EnergyTracker
try:
    from src.batch_backend import LlamaCppBatchBackend, VerifyRequest
    from src.batch_scheduler import VerificationBatchScheduler
except ImportError:  # Keeps the legacy unit-test harness importable.
    LlamaCppBatchBackend = None
    VerifyRequest = None
    VerificationBatchScheduler = None
try:
    from llama_cpp import Llama, llama_cpp
    GGUF_SUPPORT = True
except ImportError:
    GGUF_SUPPORT = False
    print("Warning: llama-cpp-python not found. GGUF model support disabled.")

# 配置
APP_PORT = 8000
POWER_SAMPLE_INTERVAL = float(os.environ.get("GPU_POWER_SAMPLE_INTERVAL", 0.005))

app = FastAPI(title="Speculative Decoding Communication Gateway")
shared_model = None
batch_scheduler = None
server_args = None

# logging setup
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("communication_service")
logger.setLevel(logging.INFO)
log_path = os.path.join(LOG_DIR, "communication_service.log")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
if not logger.handlers:
    try:
        fh = logging.FileHandler(log_path)
    except OSError:
        fh = logging.NullHandler()
    fh.setFormatter(formatter)
    logger.addHandler(fh)

consistency = True

gpu_energy_monitor = GPUEnergyMonitor(device_index=int(os.environ.get("GPU_ENERGY_DEVICE", 0)), logger=logger)

# 定义请求体模型
class InitRequest(BaseModel):
    task_id: int
    tokens: List[int]


class TaskRequest(BaseModel):
    task_id: int

class MyModel():
    def __init__(self, model_path: str, n_ctx: int):
        args = parse_arguments()
        if not GGUF_SUPPORT:
            raise RuntimeError("llama-cpp-python is required for GGUF model support.")
        self.model = Llama(
            model_path=model_path,
            n_threads=1,
            n_threads_batch=1,
            n_gpu_layers=-1,
            use_mlock=False,
            verbose=False,
            logits_all=True,
            n_ctx=n_ctx,
            seed = args.seed
        )
        self.task_id = 0

    def set_task(self, task_id: int):
        self.task_id = task_id

    def sample_and_log(self, top_k: int = 1, top_p: float = 0.95, temp: float = 0.0, task_id: int = None):
        """Call the underlying Llama.sample and log internal state for debugging determinism.

        Logs:
        - task_id (if provided)
        - model.n_tokens before sampling
        - last logits (if available) top-10 token ids and probs
        - sampling args and returned token
        """
        tid = task_id if task_id is not None else self.task_id
        try:
            logger.info(f"model.sample start: task={tid} top_k={top_k} top_p={top_p} temp={temp} n_tokens={getattr(self.model, 'n_tokens', None)}")

            last_scores = None
            try:
                # llama-cpp stores scores (logits) on the model after eval()
                if hasattr(self.model, 'scores') and len(self.model.scores) > 0:
                    last_scores = self.model.scores[-1]
            except Exception:
                last_scores = None

            if last_scores is not None:
                try:
                    probs = softmax(last_scores)
                    # get top-10 tokens for quick inspection
                    top_k_show = min(10, probs.shape[0]) if probs.ndim == 1 else 10
                    if probs.ndim == 1:
                        top_idxs = np.argsort(probs)[-top_k_show:][::-1]
                        top_list = [(int(i), float(probs[i])) for i in top_idxs]
                    else:
                        # if it's a vector per-vocab shape
                        top_idxs = np.argsort(probs)[-top_k_show:][::-1]
                        top_list = [(int(i), float(probs[i])) for i in top_idxs]
                    logger.info(f"model.sample last_scores_topk={top_list}")
                except Exception:
                    logger.info("model.sample could not compute topk from last_scores")

            # Now call the underlying sample
            sampled = self.model.sample(top_k=top_k, top_p=top_p, temp=temp)
            logger.info(f"model.sample returned: task={tid} sampled={sampled}")
            return sampled
        except Exception as e:
            logger.exception(f"model.sample failed: {e}")
            # re-raise to keep original behavior
            raise
    
    def change_task(self, task_id: int):
        self.set_task(task_id)
        self.model.reset()

class InferenceTask:
    def __init__(self, task_id: int, prefix: List[int], args):
        self.task_id = task_id
        self.prefix = prefix
        self.args = args
        # self.load_model()
        if shared_model is not None:
            shared_model.set_task(task_id)
            self.target_model = shared_model.model  # legacy serial backend
        else:
            self.target_model = None
        self.model_state = None
        self.lock = threading.RLock()
        self.n_past = 0
        self.final_token = None  # 记录上次的final_token
        # 存储累积的推测token和概率
        self.accumulated_tokens = []
        self.accumulated_probs = []
        self.prob_transport = 'full'
        self.pending_rejection = None
        self.gamma = args.gamma if hasattr(args, 'gamma') else 4
        self.max_len = args.max_tokens if hasattr(args, 'max_tokens') else 512
        self.top_k = args.top_k if hasattr(args, 'top_k') else 1
        self.top_p = args.top_p if hasattr(args, 'top_p') else 0.95
        self.temp = args.temp if hasattr(args, 'temp') else 0
        # Do NOT reseed global RNGs per task (that would make global state order-dependent).
        # Instead create task-local RNGs for deterministic behavior within this task.
        seed = args.seed if hasattr(args, 'seed') else 1234
        self.rng = np.random.default_rng(seed)
        
        self.last_verify_pass = False
        # Verification must not hold ``self.lock`` while the target model is
        # running.  Proactive requests use this condition to deposit the next
        # round and, for a waiting-NAV request, wait for its parent NAV result.
        self.verify_condition = threading.Condition(self.lock)
        self.verify_in_progress = False
        self.active_verify_round_id = None
        self.last_completed_round_id = None
        self.completed_verifications = {}
        self.proactive_buffers = {}
        self.promoted_parent_rounds = set()
        try:
            self.torch_generator = torch.Generator()
            self.torch_generator.manual_seed(seed)
        except Exception:
            self.torch_generator = None
        self.total_gpu_power_integral_joules = 0.0
        self.prompt_prefill_gpu_energy_joules = 0.0
        self.nav_gpu_energy_joules = 0.0
        self.energy_measurement_duration_seconds = 0.0
        self.prompt_prefill_energy_measurement = None
        self.nav_energy_trace = []
        self.cloud_batch_trace = []
        self.last_verify_power_integral = 0.0
        self.veridy_num = 0
        self.cache_version = 0
        self.discarded_proactive_tokens = 0
        self.reused_proactive_tokens = 0
        self.task_energy_tracker = None
        # 记录每个绝对位置使用的随机数（用于接受判定或最终token采样）
        # 结构为列表，元素是 {'pos': int, 'rand': float, 'stage': str}
        self.rand_trace: List[Dict[str, object]] = []

    def _upsert_rand_trace(self, pos: int, rand_val: float, stage: str = "verify"):
        """将 (pos, rand) 插入或覆盖到 rand_trace 列表中。
        如果列表中已存在相同 pos 的记录，则覆盖；否则追加。
        stage 用于标记来源：'verify' 或 'final_token' 等。
        """
        for i, rec in enumerate(self.rand_trace):
            try:
                if isinstance(rec, dict) and rec.get('pos') == pos:
                    self.rand_trace[i] = {'pos': pos, 'rand': float(rand_val), 'stage': stage}
                    return
            except Exception:
                continue
        self.rand_trace.append({'pos': pos, 'rand': float(rand_val), 'stage': stage})

        consistency = True

    def _energy_context(self, stage: str):
        if not gpu_energy_monitor or not gpu_energy_monitor.enabled:
            return nullcontext()
        return EnergyTracker(gpu_energy_monitor, self, stage, POWER_SAMPLE_INTERVAL, logger=logger)

    def start_energy_tracking(self):
        """Retained for edge-client compatibility; active stages track themselves."""
        self.task_energy_tracker = None

    def stop_energy_tracking(self):
        self.task_energy_tracker = None

    def record_energy_measurement(
        self,
        *,
        stage: str,
        energy_joules: float,
        duration_seconds: float,
        sample_count: int,
    ):
        measurement = {
            'stage': stage,
            'energy_joules': float(energy_joules),
            'duration_seconds': float(duration_seconds),
            'sample_count': int(sample_count),
        }
        self.total_gpu_power_integral_joules += float(energy_joules)
        self.energy_measurement_duration_seconds += float(duration_seconds)
        if stage == 'init_eval':
            self.prompt_prefill_gpu_energy_joules += float(energy_joules)
            self.prompt_prefill_energy_measurement = measurement
        elif stage == 'verify_total':
            self.nav_gpu_energy_joules += float(energy_joules)
            self.last_verify_power_integral = float(energy_joules)
            measurement['nav_index'] = len(self.nav_energy_trace)
            self.nav_energy_trace.append(measurement)
        return measurement

    def annotate_last_nav_energy(self, *, round_id, n_past, result):
        if not self.nav_energy_trace:
            return None
        measurement = self.nav_energy_trace[-1]
        measurement.update({
            'speculative_round_id': round_id,
            'n_past': int(n_past),
            'n_speculative': int(result.get('n_speculative', 0) or 0),
            'n_accepted': int(result.get('n_accepted', 0) or 0),
        })
        return dict(measurement)
        
    def load_model(self):
        self.target_model = Llama(
            model_path=self.args.target_model,
            n_threads=self.args.threads,
            n_threads_batch=self.args.threads,
            n_gpu_layers=-1,
            use_mlock=False,
            verbose=False,
            logits_all=True,
            n_ctx=self.args.ctx_size,
            seed=self.args.seed
        )
        
    def proc_prefix(self):
        shared_model.set_task(self.task_id)
        self.target_model.reset()
        with self._energy_context("init_eval"):
            self.target_model.eval(self.prefix)
        self.n_past = self.target_model.n_tokens
        self.save_model_state()
        return self.n_past == len(self.prefix)

    def save_model_state(self):
        shared_model.set_task(self.task_id)
        self.model_state = self.target_model.save_state()
        self.n_past = self.target_model.n_tokens
        return self.model_state

    def restore_model_state(self):
        shared_model.set_task(self.task_id)
        if self.model_state is None:
            self.target_model.reset()
            self.n_past = self.target_model.n_tokens
            return
        self.target_model.load_state(self.model_state)
        self.n_past = self.target_model.n_tokens

    def add_batch(self, tokens: List[int], probs: List[List[float]], index: int):
        """
        添加一批推测token和概率
        Args:
            tokens: 推测的token列表
            probs: 对应的概率分布列表
            index: 在当前推测序列中的起始索引
        """
        # 若新批次的起始位置已经有数据，截断该位置之后的累积结果
        if index < len(self.accumulated_tokens):
            self.accumulated_tokens = self.accumulated_tokens[:index]
            self.accumulated_probs = self.accumulated_probs[:index]
        elif index > len(self.accumulated_tokens):
            # 缺口使用 None 占位，保持索引一致
            gap = index - len(self.accumulated_tokens)
            self.accumulated_tokens.extend([None] * gap)
            self.accumulated_probs.extend([None] * gap)
        self.accumulated_tokens.extend(tokens)
        self.accumulated_probs.extend([np.array(p) for p in probs])
        return index + len(tokens) == len(self.accumulated_tokens)

    @staticmethod
    def _add_indexed(tokens_store, probs_store, tokens, probs, index):
        """Insert an indexed batch into an arbitrary speculative buffer."""
        if index < len(tokens_store):
            del tokens_store[index:]
            del probs_store[index:]
        elif index > len(tokens_store):
            gap = index - len(tokens_store)
            tokens_store.extend([None] * gap)
            probs_store.extend([None] * gap)
        tokens_store.extend(tokens)
        probs_store.extend([np.array(p) for p in probs])
        return index + len(tokens) == len(tokens_store)

    def buffer_proactive_batch(self, payload, tokens, probs, index):
        """Store a next-round batch without touching the active NAV buffer."""
        parent_round_id = payload.get('parent_round_id')
        round_id = payload.get('speculative_round_id')
        if parent_round_id is None and round_id is not None:
            parent_round_id = int(round_id) - 1
        if parent_round_id is None:
            return None, False

        parent_round_id = int(parent_round_id)
        expected_prefix_token = payload.get('expected_prefix_token')
        entry = self.proactive_buffers.setdefault(parent_round_id, {
            'tokens': [],
            'probs': [],
            'expected_prefix_token': expected_prefix_token,
            'round_id': round_id,
            'prefix_version': payload.get('prefix_version'),
            'prob_transport': payload.get('prob_transport', 'full'),
        })
        if (
            entry['expected_prefix_token'] != expected_prefix_token
            or entry['round_id'] != round_id
            or entry['prefix_version'] != payload.get('prefix_version')
            or entry['prob_transport'] != payload.get('prob_transport', 'full')
        ):
            return parent_round_id, False
        added_contiguously = self._add_indexed(
            entry['tokens'], entry['probs'], tokens, probs, index
        )
        return parent_round_id, added_contiguously

    def proactive_parent_status(self, parent_round_id, expected_prefix_token):
        """Return not_started, pending, valid, or invalid for a parent NAV.

        A proactive request is allowed to overtake its parent NAV on the two
        independent HTTP channels.  Such a future round must be buffered, not
        treated as stale.  A missing round is stale only when the cloud has
        already completed a later round.
        """
        if parent_round_id is None:
            return 'invalid'
        if parent_round_id == self.active_verify_round_id and self.verify_in_progress:
            return 'pending'
        result = self.completed_verifications.get(parent_round_id)
        if result is None:
            if (
                self.last_completed_round_id is None
                or int(parent_round_id) > int(self.last_completed_round_id)
            ):
                return 'not_started'
            return 'invalid'
        all_accepted = result.get('n_accepted') == result.get('n_speculative')
        prefix_matches = (
            expected_prefix_token is not None
            and result.get('final_token') is not None
            and int(expected_prefix_token) == int(result['final_token'])
        )
        return 'valid' if all_accepted and prefix_matches else 'invalid'

    def discard_proactive_buffer(self, parent_round_id):
        entry = self.proactive_buffers.pop(parent_round_id, None)
        count = len([token for token in (entry or {}).get('tokens', []) if token is not None])
        self.discarded_proactive_tokens += count
        return count

    def promote_proactive_buffer(self, parent_round_id):
        entry = self.proactive_buffers.pop(parent_round_id, None)
        if entry is None:
            return 0
        self.accumulated_tokens = list(entry['tokens'])
        self.accumulated_probs = list(entry['probs'])
        self.prob_transport = entry.get('prob_transport', 'full')
        count = len([token for token in self.accumulated_tokens if token is not None])
        self.reused_proactive_tokens += count
        self.promoted_parent_rounds.add(parent_round_id)
        return count
        
        # print(f"[DEBUG] Added batch: tokens={tokens}, index={index}, total_tokens={len([t for t in self.accumulated_tokens if t is not None])}")

    def verify_tokens(self, n_past_at_verify):
        """
        验证累积的推测token，高度优化版本
        """
        with self._energy_context("verify_total"):
            logger.info(f"verify_tokens start: task_id={self.task_id}, n_past_at_verify={n_past_at_verify}, accumulated_tokens_len={len(self.accumulated_tokens)}, n_tokens={self.target_model.n_tokens}")

            # 快速检查
            if not self.accumulated_tokens:
                logger.info(f"verify_tokens: no accumulated tokens for task {self.task_id}")
                return {'n_accepted': 0, 'n_speculative': 0, 'final_token': None, 'n_past': n_past_at_verify}
            
            # 过滤有效数据
            valid_mask = np.array([token is not None and prob is not None 
                                for token, prob in zip(self.accumulated_tokens, self.accumulated_probs)])
            
            if not np.any(valid_mask):
                logger.info(f"verify_tokens: no valid tokens after mask for task {self.task_id}")
                return {'n_accepted': 0, 'n_speculative': 0, 'final_token': None, 'n_past': n_past_at_verify}
            
            # 提取有效数据
            valid_tokens = [self.accumulated_tokens[i] for i in range(len(self.accumulated_tokens)) if valid_mask[i]]
            valid_probs = [self.accumulated_probs[i] for i in range(len(self.accumulated_probs)) if valid_mask[i]]
            
            if not valid_tokens:
                return {'n_accepted': 0, 'n_speculative': 0, 'final_token': None, 'n_past': n_past_at_verify}
            
            speculative_tokens = valid_tokens 
            prob_transport = self.prob_transport
            if prob_transport == 'lazy_distribution':
                draft_token_probs = np.asarray(
                    [float(np.asarray(probability)) for probability in valid_probs],
                    dtype=np.float64,
                )
                draft_probs = None
            elif prob_transport == 'full':
                draft_probs = np.stack(valid_probs)  # 假设所有probs形状相同
            else:
                raise ValueError(f"unknown probability transport: {prob_transport}")
            n_speculative = len(speculative_tokens)
            logger.info(f"task={self.task_id} n_speculative={n_speculative} speculative_tokens={self.target_model.detokenize(speculative_tokens).decode('utf-8', 'ignore')}")
            logger.info(
                "task=%s draft_probability_transport=%s",
                self.task_id,
                prob_transport,
            )
            
            if self.final_token is not None:
                final_len = 1
            else:
                final_len = 0
            
            if n_past_at_verify - final_len < self.target_model.n_tokens:
                self.final_token = None
                self.target_model.n_tokens = n_past_at_verify
                logger.info(f"覆盖之前的记录, n_past={n_past_at_verify}, n_tokens={self.target_model.n_tokens}")

            # 评估
            eval_tokens = [self.final_token] + speculative_tokens if self.final_token is not None else speculative_tokens
            self.target_model.eval(eval_tokens)
            
            # 获取目标概率
            target_scores = self.target_model.scores[n_past_at_verify-1 : n_past_at_verify-1 + n_speculative]
            target_probs = softmax(target_scores)
            logger.info(f"task={self.task_id} target_probs_shape={target_probs.shape}")
            
            # 向量化计算概率比值
            EPSILON = 1e-9
            target_token_probs = target_probs[np.arange(n_speculative), speculative_tokens]
            if prob_transport == 'full':
                draft_token_probs = draft_probs[
                    np.arange(n_speculative), speculative_tokens
                ]
            p_ratios = target_token_probs / (draft_token_probs + EPSILON)
            # p_ratios = np.round(p_ratios, decimals=2)
            logger.info(f"task={self.task_id} scores={target_scores[np.arange(n_speculative), speculative_tokens].tolist()}")
            logger.info(f"task={self.task_id} target_token_probs={target_token_probs.tolist()}")
            logger.info(f"task={self.task_id} draft_token_probs={draft_token_probs.tolist()}")
            logger.info(f"task={self.task_id} p_ratios={p_ratios.tolist()}")

            if n_past_at_verify != self.target_model.n_tokens:
                consistency = False
            # 逐个验证直到遇到拒绝
            n_accepted = 0
            for i in range(n_speculative):
                global_idx = n_past_at_verify + i  # 该 token 在整条序列里的位置（0 基）
                rng = random.Random(self.args.seed + global_idx)
                rand_val = rng.random()
                # 记录 (位置, 随机数)
                self._upsert_rand_trace(global_idx, rand_val, stage="verify")
                accept = (p_ratios[i] >= 1.0) or (rand_val < float(p_ratios[i]))
                logger.info(f"task={self.task_id} idx={i} token={speculative_tokens[i]} p_ratio={float(p_ratios[i])} rand={rand_val} accept={accept} seed={self.args.seed + global_idx} ntokens={self.target_model.n_tokens}")
                if accept:
                    n_accepted += 1
                else:
                    break
            
            # 更新状态
            self.target_model.n_tokens = n_past_at_verify + n_accepted
            
            # 计算最终token
            if n_accepted < n_speculative:
                if prob_transport == 'lazy_distribution':
                    verification_id = uuid.uuid4().hex
                    self.pending_rejection = {
                        'verification_id': verification_id,
                        'target_probs': target_probs[n_accepted].copy(),
                        'draft_token_prob': float(draft_token_probs[n_accepted]),
                        'draft_token': int(speculative_tokens[n_accepted]),
                        'n_accepted': n_accepted,
                        'n_speculative': n_speculative,
                        'accepted_n_past': n_past_at_verify + n_accepted,
                    }
                    final_token = None
                else:
                    diff_probs = target_probs[n_accepted] - draft_probs[n_accepted]
                    logger.info(f"task={self.task_id} n_accepted={n_accepted} computing final_token from diff_probs")
                    try:
                        seed_for_sample = self.args.seed if hasattr(self.args, 'seed') else None
                    except Exception:
                        seed_for_sample = None
                    logger.info(f"task={self.task_id} sample_seed={seed_for_sample}")
                    # 为最终 token 的位置生成并记录一个随机数键（与 verify 相同位置规则）
                    final_token = sample(max_fn(diff_probs), 1, seed=seed_for_sample)
                    logger.info(f"task={self.task_id} sampled final_token={final_token}")
            else:
                # Use shared_model wrapper to log internal model sampling behavior
                try:
                    # full-accept 的情况下，同样记录一个位置键，便于统一对齐（不影响采样）
                    final_token = shared_model.sample_and_log(
                        top_k=self.top_k, top_p=self.top_p, temp=self.temp, task_id=self.task_id
                    )
                    logger.info(f"task={self.task_id} used model.sample final_token={final_token}")
                except Exception:
                    # Fallback to calling the model directly (preserve previous behavior)
                    final_token = self.target_model.sample(top_k=self.top_k, top_p=self.top_p, temp=self.temp)
                    logger.info(f"task={self.task_id} used model.sample (direct fallback) final_token={final_token}")
            
            # 更新并返回
            new_n_past = self.target_model.n_tokens + (1 if final_token is not None else 0)
            self.final_token = final_token
            self.reset_accumulated()
            logger.info(f"verify_tokens end: task={self.task_id} n_accepted={n_accepted} n_speculative={n_speculative} final_token={final_token} new_n_past={new_n_past}")

            if self.pending_rejection is not None:
                return {
                    'status': 'needs_full_probs',
                    'verification_id': self.pending_rejection['verification_id'],
                    'rejected_index': n_accepted,
                    'n_accepted': n_accepted,
                    'n_speculative': n_speculative,
                    'final_token': None,
                    'n_past': new_n_past,
                    'gpu_power_integral': getattr(self, 'last_verify_power_integral', 0.0),
                }
            if n_accepted == n_speculative:
                self.last_verify_pass = True
            else:
                self.last_verify_pass = False
                self.reset_accumulated()
            self.cache_version += 1

            return {
                'n_accepted': n_accepted,
                'n_speculative': n_speculative,
                'final_token': final_token,
                'n_past': new_n_past,
                'cache_version': self.cache_version,
                'gpu_power_integral': getattr(self, 'last_verify_power_integral', 0.0)
            }

    def resolve_rejection(self, verification_id: str, draft_probs):
        pending = self.pending_rejection
        if pending is None:
            raise ValueError("task has no pending rejection")
        if pending['verification_id'] != verification_id:
            raise ValueError("verification_id does not match pending rejection")
        draft_probs = np.asarray(draft_probs, dtype=np.float64).reshape(-1)
        target_probs = np.asarray(pending['target_probs'], dtype=np.float64)
        if draft_probs.shape != target_probs.shape:
            raise ValueError(
                f"draft probability shape {draft_probs.shape} does not match "
                f"target shape {target_probs.shape}"
            )
        token = int(pending['draft_token'])
        if not np.isclose(
            draft_probs[token],
            float(pending['draft_token_prob']),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError("resolved draft probability does not match proposal scalar")
        seed_for_sample = self.args.seed if hasattr(self.args, 'seed') else None
        final_token = sample(
            max_fn(target_probs - draft_probs), 1, seed=seed_for_sample
        )
        self.final_token = final_token
        self.pending_rejection = None
        self.last_verify_pass = False
        self.cache_version += 1
        return {
            'status': 'resolved',
            'n_accepted': int(pending['n_accepted']),
            'n_speculative': int(pending['n_speculative']),
            'final_token': int(final_token),
            'n_past': int(pending['accepted_n_past']) + 1,
            'cache_version': self.cache_version,
            'gpu_power_integral': getattr(self, 'last_verify_power_integral', 0.0),
        }
    
    def reset_accumulated(self):
        """重置累积的数据"""
        self.accumulated_tokens = []
        self.accumulated_probs = []

# 全局任务字典
active_tasks: Dict[int, InferenceTask] = {}
active_tasks_lock = threading.RLock()
model_lock = threading.RLock()


def _runtime_args():
    return server_args if server_args is not None else parse_arguments()


def _batch_enabled():
    return batch_scheduler is not None


def _finalize_verification(task, round_id, result):
    """Commit a completed NAV result and release/discard proactive work."""
    task.completed_verifications[round_id] = dict(result)
    task.last_completed_round_id = round_id
    task.last_verify_pass = (
        result.get('n_accepted') == result.get('n_speculative')
    )
    task.final_token = result.get('final_token')
    proactive_entry = task.proactive_buffers.get(round_id)
    if proactive_entry is None:
        return
    status = task.proactive_parent_status(
        round_id, proactive_entry.get('expected_prefix_token')
    )
    if status == 'pending':
        all_accepted = result.get('n_accepted') == result.get('n_speculative')
        prefix_matches = (
            proactive_entry.get('expected_prefix_token') is not None
            and result.get('final_token') is not None
            and int(proactive_entry['expected_prefix_token'])
            == int(result['final_token'])
        )
        status = 'valid' if all_accepted and prefix_matches else 'invalid'
    if status == 'valid':
        task.promote_proactive_buffer(round_id)
    else:
        task.discard_proactive_buffer(round_id)


def _decode_residual_distribution(payload):
    if payload.get('prob_dtype') != 'float32':
        raise ValueError("lazy residual probability dtype must be float32")
    vocab_size = int(payload.get('vocab_size', 0))
    raw = payload.get('prob_bytes')
    if vocab_size <= 0 or not isinstance(raw, (bytes, bytearray)):
        raise ValueError("lazy residual payload is missing probability bytes")
    probabilities = np.frombuffer(raw, dtype='<f4')
    if probabilities.size != vocab_size:
        raise ValueError(
            f"lazy residual has {probabilities.size} values, expected {vocab_size}"
        )
    return probabilities.copy()


def handle_init_request(request: InitRequest):
    args = _runtime_args()
    task = InferenceTask(request.task_id, request.tokens, args)
    init_result = None
    if _batch_enabled():
        init_result = batch_scheduler.initialize(request.task_id, request.tokens)
        task.n_past = int(init_result['n_past'])
        prefill_energy = float(init_result.get('prefill_gpu_power_integral', 0.0))
        task.record_energy_measurement(
            stage='init_eval',
            energy_joules=prefill_energy,
            duration_seconds=float(init_result.get('prefill_seconds', 0.0)),
            sample_count=0,
        )
        task.cloud_batch_trace.append({
            key: init_result.get(key) for key in (
                'batch_stage', 'batch_id', 'actual_batch_size',
                'actual_batch_tokens', 'batch_queue_seconds',
                'prefill_seconds', 'evaluated_tokens', 'seq_id',
                'prefill_gpu_power_integral', 'batch_energy_joules',
                'energy_allocation',
            )
        })
        task.cloud_batch_trace.append({
            key: init_result.get(key) for key in (
                'batch_stage', 'batch_id', 'actual_batch_size',
                'actual_batch_tokens', 'batch_queue_seconds',
                'prefill_seconds', 'evaluated_tokens', 'seq_id',
                'prefill_gpu_power_integral', 'batch_energy_joules',
                'energy_allocation',
            )
        })
    else:
        with task.lock:
            with model_lock:
                success = task.proc_prefix()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process prefix tokens.")

    with active_tasks_lock:
        active_tasks[request.task_id] = task

    response = {'init': 'success', 'n_past': task.n_past}
    if init_result is not None:
        response.update({'backend': 'batched', 'seq_id': init_result.get('seq_id')})
    return response


def handle_start_request(request: TaskRequest):
    with active_tasks_lock:
        task = active_tasks.get(request.task_id)
    if task is None:
        raise HTTPException(status_code=400, detail="Task not found or not initialized.")
    with task.lock:
        task.start_energy_tracking()
    return {
        'status': 'started',
        'task_id': request.task_id,
        'energy_scope': 'cloud_gpu_prompt_prefill_plus_nav_compute',
    }

def handle_propose_payload(payload):
    task_id = payload.get('task_id')
    with active_tasks_lock:
        task = active_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=400, detail="Task not found or not initialized.")

    tokens = payload.get('tokens', [])
    probs = payload.get('probs', [])
    prob_transport = payload.get('prob_transport', 'full')
    if prob_transport not in {'full', 'lazy_distribution'}:
        raise HTTPException(status_code=400, detail="Unknown probability transport.")
    should_verify = payload.get('should_verify', False)
    n_past_at_receive = payload.get('n_past', task.n_past)

    is_proactive = payload.get('type') == 'propose_waiting'
    round_id = payload.get('speculative_round_id')

    if is_proactive:
        with task.verify_condition:
            index = payload.get('index', 0)
            parent_round_id = payload.get('parent_round_id')
            if parent_round_id is None and round_id is not None:
                parent_round_id = int(round_id) - 1
            if parent_round_id is not None:
                parent_round_id = int(parent_round_id)
            expected_prefix_token = payload.get('expected_prefix_token')
            parent_status = (
                task.proactive_parent_status(parent_round_id, expected_prefix_token)
                if parent_round_id is not None else None
            )
            if (
                parent_status == 'valid'
                and parent_round_id in task.promoted_parent_rounds
            ):
                add_result = task.add_batch(tokens, probs, index)
                task.reused_proactive_tokens += len(tokens)
            else:
                parent_round_id, add_result = task.buffer_proactive_batch(
                    payload, tokens, probs, index
                )

            # Backward-compatible path for old clients without round metadata.
            # New clients always send parent_round_id and take the concurrent path.
            if parent_round_id is None:
                valid = (
                    task.last_verify_pass
                    and expected_prefix_token is not None
                    and task.final_token is not None
                    and int(expected_prefix_token) == int(task.final_token)
                )
                if not valid:
                    task.discarded_proactive_tokens += len(tokens)
                    return {
                        'status': 'discarded_stale_proactive_batch',
                        'n_past': task.n_past,
                        'cache_version': task.cache_version,
                        'discarded_proactive_tokens': len(tokens),
                    }
                task.reused_proactive_tokens += len(tokens)
                task.add_batch(tokens, probs, index)
            else:
                status = task.proactive_parent_status(parent_round_id, expected_prefix_token)
                if status in {'not_started', 'pending'} and not should_verify:
                    return {
                        'status': 'buffered_pending_nav',
                        'parent_status': status,
                        'n_past': task.n_past,
                        'parent_round_id': parent_round_id,
                        'total_buffered': len([
                            token for token in task.proactive_buffers[parent_round_id]['tokens']
                            if token is not None
                        ]),
                        'add_result': add_result,
                    }
                # A waiting-NAV request may reach the cloud before its parent
                # NAV.  Wait for the parent to arrive and finish, releasing the
                # condition so the parent request can make progress.
                while status in {'not_started', 'pending'}:
                    task.verify_condition.wait()
                    status = task.proactive_parent_status(parent_round_id, expected_prefix_token)
                if status != 'valid':
                    discarded = task.discard_proactive_buffer(parent_round_id)
                    return {
                        'status': 'discarded_stale_proactive_batch',
                        'reason': 'parent_nav_invalid',
                        'parent_round_id': parent_round_id,
                        'speculative_round_id': round_id,
                        'n_past': task.n_past,
                        'cache_version': task.cache_version,
                        'discarded_proactive_tokens': discarded,
                    }
                # The completing NAV normally promotes batches that arrived while
                # it was running.  A batch arriving just after completion is
                # promoted here instead.
                if parent_round_id in task.proactive_buffers:
                    task.promote_proactive_buffer(parent_round_id)

            if not should_verify:
                return {
                    'status': 'accumulated',
                    'n_past': task.n_past,
                    'total_accumulated': len([
                        token for token in task.accumulated_tokens if token is not None
                    ]),
                    'add_result': add_result,
                }

    with task.verify_condition:
        index = payload.get(
            'index', len([t for t in task.accumulated_tokens if t is not None])
        )
        if not is_proactive:
            task.prob_transport = prob_transport
            task.add_batch(tokens, probs, index)
        if not should_verify:
            return {
                'status': 'accumulated',
                'n_past': task.n_past,
                'total_accumulated': len([
                    token for token in task.accumulated_tokens if token is not None
                ]),
                'add_result': True,
            }

        while task.verify_in_progress:
            task.verify_condition.wait()
        task.verify_in_progress = True
        task.active_verify_round_id = int(round_id) if round_id is not None else task.cache_version
        active_round_id = task.active_verify_round_id
        if _batch_enabled():
            verify_prob_transport = task.prob_transport
            verify_tokens = [
                int(token) for token in task.accumulated_tokens if token is not None
            ]
            verify_probs = [
                np.asarray(probability, dtype=np.float64)
                for token, probability in zip(
                    task.accumulated_tokens, task.accumulated_probs
                )
                if token is not None
            ]
            task.reset_accumulated()
        # Wake a proactive waiting-NAV request that arrived before its parent.
        # It will observe ``pending`` and continue waiting for completion.
        task.verify_condition.notify_all()

    # Do not hold task.lock here: proactive HTTP requests must remain able to
    # populate task.proactive_buffers while target-model verification runs.
    try:
        if _batch_enabled():
            request = VerifyRequest(
                task_id=task.task_id,
                n_past=int(n_past_at_receive),
                tokens=verify_tokens,
                draft_probs=verify_probs,
                seed=int(getattr(task.args, 'seed', 1234)),
                temp=float(task.temp),
                top_k=int(task.top_k),
                top_p=float(task.top_p),
                prob_transport=verify_prob_transport,
            )
            result = batch_scheduler.submit(request)
            task.n_past = int(result['n_past'])
            if result.get('status') != 'needs_full_probs':
                task.cache_version += 1
                result['cache_version'] = task.cache_version
            task.veridy_num += 1
            nav_energy = task.record_energy_measurement(
                stage='verify_total',
                energy_joules=float(result.get('gpu_power_integral', 0.0)),
                duration_seconds=float(result.get('batch_decode_seconds', 0.0)),
                sample_count=0,
            )
            nav_energy.update({
                'speculative_round_id': active_round_id,
                'n_past': int(n_past_at_receive),
                'n_speculative': int(result.get('n_speculative', 0)),
                'n_accepted': int(result.get('n_accepted', 0)),
                'batch_id': result.get('batch_id'),
                'actual_batch_size': result.get('actual_batch_size'),
                'actual_batch_tokens': result.get('actual_batch_tokens'),
                'energy_allocation': result.get('energy_allocation'),
            })
            result['nav_energy_measurement'] = dict(nav_energy)
            task.cloud_batch_trace.append({
                key: result.get(key) for key in (
                    'batch_id', 'actual_batch_size', 'actual_batch_tokens',
                    'batch_queue_seconds', 'batch_decode_seconds',
                    'evaluated_tokens', 'seq_id', 'gpu_power_integral',
                    'batch_energy_joules', 'energy_allocation',
                )
            })
        else:
            with model_lock:
                nav_measurement_count_before = len(task.nav_energy_trace)
                task.restore_model_state()
                result = task.verify_tokens(n_past_at_receive)
                task.save_model_state()
                task.veridy_num += 1
                nav_energy = (
                    task.annotate_last_nav_energy(
                        round_id=active_round_id,
                        n_past=n_past_at_receive,
                        result=result,
                    )
                    if len(task.nav_energy_trace) > nav_measurement_count_before
                    else None
                )
                if nav_energy is not None:
                    result['gpu_power_integral'] = nav_energy['energy_joules']
                    result['nav_energy_measurement'] = nav_energy
        if result.get('status') == 'needs_full_probs':
            if task.pending_rejection is None:
                task.pending_rejection = {
                    'verification_id': result['verification_id'],
                }
            task.pending_rejection['round_id'] = int(active_round_id)
    finally:
        with task.verify_condition:
            if (
                'result' in locals()
                and result.get('status') != 'needs_full_probs'
            ):
                _finalize_verification(task, active_round_id, result)
            task.verify_in_progress = False
            task.active_verify_round_id = None
            task.verify_condition.notify_all()
    return result


def handle_resolve_rejection_payload(payload):
    task_id = payload.get('task_id')
    verification_id = payload.get('verification_id')
    with active_tasks_lock:
        task = active_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=400, detail="Task not found or not initialized.")
    try:
        draft_probs = _decode_residual_distribution(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with task.verify_condition:
        pending = task.pending_rejection
        if _batch_enabled():
            # The backend owns the target row in batched mode.  The task keeps
            # only the round identity needed by the HTTP/proactive state machine.
            pending = getattr(task, 'pending_rejection', None)
        if pending is None:
            raise HTTPException(status_code=409, detail="Task has no pending rejection.")
        round_id = int(pending['round_id'])

    try:
        if _batch_enabled():
            result = batch_scheduler.resolve_rejection(
                int(task_id), str(verification_id), draft_probs
            )
            task.cache_version += 1
            result['cache_version'] = task.cache_version
        else:
            result = task.resolve_rejection(str(verification_id), draft_probs)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    with task.verify_condition:
        task.pending_rejection = None
        task.n_past = int(result['n_past'])
        _finalize_verification(task, round_id, result)
        task.verify_condition.notify_all()
    return result


def handle_exit_payload(payload):
    response = {"status": "exited"}

    if payload.get('type') == 'exit':
        task_id = payload.get('task_id')
        response['task_id'] = task_id
        with active_tasks_lock:
            task = active_tasks.pop(task_id, None) if task_id is not None else None
        if task is not None:
            if _batch_enabled():
                batch_scheduler.close_session(task_id)
            task.stop_energy_tracking()
            energy_available = bool(
                gpu_energy_monitor and gpu_energy_monitor.enabled
            )
            power_int = (
                task.total_gpu_power_integral_joules
                if energy_available
                else None
            )
            response['gpu_power_integral_joules'] = power_int
            response['model_energy_joules'] = power_int
            response['prompt_prefill_gpu_energy_joules'] = (
                task.prompt_prefill_gpu_energy_joules
                if energy_available
                else None
            )
            response['nav_gpu_energy_joules'] = (
                task.nav_gpu_energy_joules
                if energy_available
                else None
            )
            response['prompt_prefill_energy_measurement'] = (
                dict(task.prompt_prefill_energy_measurement)
                if task.prompt_prefill_energy_measurement is not None
                else None
            )
            response['nav_energy_trace'] = [
                dict(measurement) for measurement in task.nav_energy_trace
            ]
            response['cloud_batch_trace'] = [
                dict(measurement) for measurement in task.cloud_batch_trace
            ]
            response['energy_measurement_duration_seconds'] = (
                task.energy_measurement_duration_seconds
                if energy_available
                else None
            )
            response['energy_measurement_available'] = energy_available
            response['energy_scope'] = 'cloud_gpu_prompt_prefill_plus_nav_compute'
            response['energy_source'] = 'nvml_gpu_board_power'
            response['energy_sample_interval_seconds'] = POWER_SAMPLE_INTERVAL
            response['energy_included_stages'] = [
                'cloud_prompt_prefill',
                'target_model_nav_compute',
            ]
            response['energy_excluded_stages'] = [
                'between_nav_gpu_idle',
                'edge_draft_wait',
                'network_transfer',
                'proactive_wait_and_transfer',
                'model_load',
                'model_state_restore_and_save',
            ]
            if _batch_enabled():
                response['energy_excluded_stages'].remove(
                    'model_state_restore_and_save'
                )
                response['cloud_batch_scheduler'] = batch_scheduler.snapshot()
            response['verify_num'] = task.veridy_num
            response['cache_version'] = task.cache_version
            response['discarded_proactive_tokens'] = task.discarded_proactive_tokens
            response['reused_proactive_tokens'] = task.reused_proactive_tokens
            logger.info(
                "task=%s active_compute_gpu_energy=%sJ prefill=%.6fJ nav=%.6fJ (final)",
                task_id,
                f"{power_int:.6f}" if power_int is not None else "unavailable",
                task.prompt_prefill_gpu_energy_joules,
                task.nav_gpu_energy_joules,
            )

    return response

@app.post("/init")
async def init(request: InitRequest):  # 直接使用 Pydantic 模型
    from anyio.to_thread import run_sync

    return await run_sync(handle_init_request, request)


@app.post("/start")
async def start(request: TaskRequest):
    from anyio.to_thread import run_sync

    return await run_sync(handle_start_request, request)

@app.post("/delay")
async def delay(request: Request):
    # 关键：强制读取整个请求体
    body = await request.body()
    receive_time = time.time()
    return {
        "receive_time": receive_time,
        "body_size_bytes": len(body)
    }

@app.post("/propose")
async def propose(request: Request):
    raw_body = await request.body()
    
    try:
        payload = msgpack.unpackb(raw_body, raw=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid msgpack data: {str(e)}")

    from anyio.to_thread import run_sync

    return await run_sync(handle_propose_payload, payload)


@app.post("/resolve_rejection")
async def resolve_rejection(request: Request):
    raw_body = await request.body()
    try:
        payload = msgpack.unpackb(raw_body, raw=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid msgpack data: {str(exc)}"
        ) from exc

    from anyio.to_thread import run_sync

    return await run_sync(handle_resolve_rejection_payload, payload)

@app.post("/exit")
async def exit_task(request: Request):
    raw_body = await request.body()
    payload = msgpack.unpackb(raw_body, raw=False)

    from anyio.to_thread import run_sync

    return await run_sync(handle_exit_payload, payload)

@app.get("/health")
async def health():
    response = {
        "status": "running",
        "backend": "batched" if _batch_enabled() else "serial",
        "active_tasks": len(active_tasks),
    }
    if _batch_enabled():
        response["batch_scheduler"] = batch_scheduler.snapshot()
    return response

@app.get("/")
async def root():
    return {"message": "Speculative Decoding Communication Gateway is running"}

if __name__ == "__main__":
    import uvicorn
    server_args = parse_arguments()
    seed_everything(server_args.seed)
    if server_args.backend == 'batched':
        if LlamaCppBatchBackend is None or VerificationBatchScheduler is None:
            raise RuntimeError("batched backend modules could not be imported")
        backend = LlamaCppBatchBackend(
            model_path=server_args.target_model,
            max_sequences=server_args.max_sequences,
            context_tokens_per_sequence=server_args.ctx_size,
            decode_batch_tokens=server_args.batch_size,
            physical_batch_tokens=server_args.ubatch_size,
            threads=server_args.threads,
            flash_attention=not server_args.disable_flash_attention,
        )
        batch_scheduler = VerificationBatchScheduler(
            backend,
            max_batch_clients=server_args.max_sequences,
            max_batch_tokens=server_args.batch_size,
            batch_wait_ms=server_args.batch_wait_ms,
            request_timeout_s=server_args.batch_request_timeout_s,
            energy_context_factory=lambda sink, stage: EnergyTracker(
                gpu_energy_monitor,
                sink,
                stage,
                POWER_SAMPLE_INTERVAL,
                logger=logger,
            ) if gpu_energy_monitor.enabled else nullcontext(),
        )
    else:
        shared_model = MyModel(server_args.target_model, server_args.ctx_size)
    print(
        f"🚀 启动通信服务，监听端口 {server_args.port}，"
        f"backend={server_args.backend}..."
    )
    try:
        uvicorn.run(app, host="0.0.0.0", port=server_args.port, workers=1)
    finally:
        if batch_scheduler is not None:
            batch_scheduler.shutdown()

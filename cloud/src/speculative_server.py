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
import torch
from contextlib import nullcontext
from src.util import seed_everything, parse_arguments, softmax, max_fn, sample, GPUEnergyMonitor, EnergyTracker
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

# logging setup
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("communication_service")
logger.setLevel(logging.INFO)
log_path = os.path.join(LOG_DIR, "communication_service.log")
fh = logging.FileHandler(log_path)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
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
        shared_model.set_task(task_id)
        self.target_model = shared_model.model  # 使用共享模型实例
        self.model_state = None
        self.lock = threading.RLock()
        self.n_past = 0
        self.final_token = None  # 记录上次的final_token
        # 存储累积的推测token和概率
        self.accumulated_tokens = []
        self.accumulated_probs = []
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
        if self.task_energy_tracker is not None:
            return nullcontext()
        if not gpu_energy_monitor or not gpu_energy_monitor.enabled:
            return nullcontext()
        return EnergyTracker(gpu_energy_monitor, self, stage, POWER_SAMPLE_INTERVAL, logger=logger)

    def start_energy_tracking(self):
        if gpu_energy_monitor and gpu_energy_monitor.enabled and self.task_energy_tracker is None:
            self.task_energy_tracker = EnergyTracker(
                gpu_energy_monitor, self, "task_total", POWER_SAMPLE_INTERVAL, logger=logger
            )
            self.task_energy_tracker.__enter__()

    def stop_energy_tracking(self):
        tracker, self.task_energy_tracker = self.task_energy_tracker, None
        if tracker is not None:
            tracker.__exit__(None, None, None)
        
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
        })
        if (
            entry['expected_prefix_token'] != expected_prefix_token
            or entry['round_id'] != round_id
            or entry['prefix_version'] != payload.get('prefix_version')
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
            draft_probs = np.stack(valid_probs)  # 假设所有probs形状相同
            n_speculative = len(speculative_tokens)
            logger.info(f"task={self.task_id} n_speculative={n_speculative} speculative_tokens={self.target_model.detokenize(speculative_tokens).decode('utf-8', 'ignore')}")
            logger.info(f"task={self.task_id} draft_probs_shape={draft_probs.shape}")
            
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
            draft_token_probs = draft_probs[np.arange(n_speculative), speculative_tokens]
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
            new_n_past = self.target_model.n_tokens + 1
            self.final_token = final_token
            self.reset_accumulated()
            logger.info(f"verify_tokens end: task={self.task_id} n_accepted={n_accepted} n_speculative={n_speculative} final_token={final_token} new_n_past={new_n_past}")

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
    
    def reset_accumulated(self):
        """重置累积的数据"""
        self.accumulated_tokens = []
        self.accumulated_probs = []

# 全局任务字典
active_tasks: Dict[int, InferenceTask] = {}
active_tasks_lock = threading.RLock()
model_lock = threading.RLock()


def handle_init_request(request: InitRequest):
    args = parse_arguments()
    task = InferenceTask(request.task_id, request.tokens, args)
    with task.lock:
        with model_lock:
            success = task.proc_prefix()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process prefix tokens.")

    with active_tasks_lock:
        active_tasks[request.task_id] = task

    return {'init': 'success', 'n_past': task.n_past}


def handle_start_request(request: TaskRequest):
    with active_tasks_lock:
        task = active_tasks.get(request.task_id)
    if task is None:
        raise HTTPException(status_code=400, detail="Task not found or not initialized.")
    with task.lock:
        task.start_energy_tracking()
    return {'status': 'started', 'task_id': request.task_id}

def handle_propose_payload(payload):
    task_id = payload.get('task_id')
    with active_tasks_lock:
        task = active_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=400, detail="Task not found or not initialized.")

    tokens = payload.get('tokens', [])
    probs = payload.get('probs', [])
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
        # Wake a proactive waiting-NAV request that arrived before its parent.
        # It will observe ``pending`` and continue waiting for completion.
        task.verify_condition.notify_all()

    # Do not hold task.lock here: proactive HTTP requests must remain able to
    # populate task.proactive_buffers while target-model verification runs.
    try:
        with model_lock:
            task.restore_model_state()
            result = task.verify_tokens(n_past_at_receive)
            task.save_model_state()
            task.veridy_num += 1
    finally:
        with task.verify_condition:
            if 'result' in locals():
                task.completed_verifications[active_round_id] = dict(result)
                task.last_completed_round_id = active_round_id
                task.last_verify_pass = (
                    result.get('n_accepted') == result.get('n_speculative')
                )
                task.final_token = result.get('final_token')
                proactive_entry = task.proactive_buffers.get(active_round_id)
                if proactive_entry is not None:
                    status = task.proactive_parent_status(
                        active_round_id, proactive_entry.get('expected_prefix_token')
                    )
                    # Temporarily clear the in-progress flag so status consults
                    # the completed result just recorded above.
                    if status == 'pending':
                        all_accepted = result.get('n_accepted') == result.get('n_speculative')
                        prefix_matches = (
                            proactive_entry.get('expected_prefix_token') is not None
                            and result.get('final_token') is not None
                            and int(proactive_entry['expected_prefix_token']) == int(result['final_token'])
                        )
                        status = 'valid' if all_accepted and prefix_matches else 'invalid'
                    if status == 'valid':
                        task.promote_proactive_buffer(active_round_id)
                    else:
                        task.discard_proactive_buffer(active_round_id)
            task.verify_in_progress = False
            task.active_verify_round_id = None
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
            task.stop_energy_tracking()
            power_int = task.total_gpu_power_integral_joules
            response['gpu_power_integral_joules'] = power_int
            response['verify_num'] = task.veridy_num
            response['cache_version'] = task.cache_version
            response['discarded_proactive_tokens'] = task.discarded_proactive_tokens
            response['reused_proactive_tokens'] = task.reused_proactive_tokens
            logger.info("task=%s gpu_power_integral_total=%.6fJ (final)", task_id, power_int)

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

@app.post("/exit")
async def exit_task(request: Request):
    raw_body = await request.body()
    payload = msgpack.unpackb(raw_body, raw=False)

    from anyio.to_thread import run_sync

    return await run_sync(handle_exit_payload, payload)

@app.get("/health")
async def health():
    return {"status": "running", "backend": "inference_service", "active_tasks": len(active_tasks)}

@app.get("/")
async def root():
    return {"message": "Speculative Decoding Communication Gateway is running"}

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 启动通信服务，监听端口 {APP_PORT}...")
    args = parse_arguments()
    seed_everything(args.seed)
    shared_model = MyModel(args.target_model, 16384)
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, workers=1)

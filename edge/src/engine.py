import os
import json
import torch
import transformers
import warnings
transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from abc import ABC, abstractmethod
from .util import seed_everything, softmax, strategy2exp
from .merge import dynamic_token_scheduling_dp
import time
import numpy as np
from typing import List, Dict, Tuple
import msgpack
from .comm import BandwidthSender
# For GGUF model support
try:
    from llama_cpp import Llama
    GGUF_SUPPORT = True
except ImportError:
    GGUF_SUPPORT = False
    print("Warning: llama-cpp-python not found. GGUF model support disabled.")

# ========== 配置 ==========
# URL = "http://39.102.209.27:6001"  # 你的云端 FastAPI 地址
# URL = "http://106.63.100.63:30007"
URL = "http://115.190.90.101:1597"

INIT_ENDPOINT = f"{URL}/init"
PROPOSE_ENDPOINT = f"{URL}/propose"
EXIT_ENDPOINT = f"{URL}/exit"

max_probs = []

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
        self.merge_policy: str = getattr(args, 'merge_policy', 'dp')
        self.verify_thresh_single = getattr(args, 'verify_thresh_single', 0.94)
        self.verify_thresh_multi = getattr(args, 'verify_thresh_multi', 0.9)
        
        self.multiply_times: float = getattr(args, 'multiply_times', 0.7)
        self.algorithm = getattr(args, 'algorithm', "vanilla")
        self.result_tag: str = getattr(args, 'result_tag', "")
        self.start_index_of_sample = getattr(args, 'start_index_of_sample', 0)
        self.end_index_of_sample = getattr(args, 'end_index_of_sample', 1)
        self._token_time_ref: float = 0.0
        self._token_durations: List[float] = []
        # self.exp_name = strategy2exp(self.verify_strategy)
        self.exp_name = os.path.join(os.getcwd(), 'exp', "exp__gsm", self.args.dataset, self.algorithm)
        print(self.exp_name)
        os.makedirs(self.exp_name, exist_ok=True)
        if self.algorithm == "vanilla" or self.algorithm == "hsl":
            self.send_while_generating = False
        else:
            self.send_while_generating = True
        

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
        self._token_time_ref = time.time()

        self.verify_thresh_single = self.args.verify_thresh_single
        self.verify_thresh_multi = self.args.verify_thresh_multi

        self.alpha: float = getattr(self.args, 'init_alpha', 0.01)

        self.color_print("[Edge] 工作进程启动，加载模型...", 2)
        start_time = time.time()
        if not self.draft_model:
            self.draft_model = Llama(
                model_path=self.args.draft_model, n_gpu_layers=0, n_threads=self.args.threads,
                verbose=False, logits_all=True, n_ctx=self.args.ctx_size,
            )
        end_time = time.time()
        self.color_print(f"[Edge] 模型加载完成，耗时: {end_time - start_time:.2f} 秒", 5)

        self.sender = BandwidthSender(
            bandwidth_MBps=self.bandwidth_MBps,
            base_latency=self.C,
            use_env_proxy=getattr(self.args, "use_env_proxy", False),
        )

    def _resolve_merge_plan(self) -> List[int]:
        if self.merge_policy == "immediate":
            return [1] * 40
        if self.merge_policy == "no_early":
            return [100]

        batches, _ = dynamic_token_scheduling_dp(
            [self.args.default_token_compute] * self.gamma,
            self.C,
            (self.args.token_size_MB / self.bandwidth_MBps) if self.bandwidth_MBps else 0.0,
        )
        return [len(batch) for batch in batches if batch] or [self.gamma * 4]

    def _record_token_time(self, token_count: int) -> None:
        if token_count <= 0:
            return
        now = time.time()
        elapsed = max(0.0, now - self._token_time_ref)
        per_token = elapsed / token_count
        self._token_durations.extend([per_token] * token_count)
        self._token_time_ref = now

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
    
    def update_thresh(self, multiply_times, n_accepted, n_all):
        if n_all == n_accepted:
            self.alpha *= multiply_times
        else:
            temp = (self.accumulated_probs ** ((n_all - n_accepted) / n_all))
            if temp > 0:
                self.alpha /= temp
            else:
                self.alpha /= 0.5
            self.alpha = min(1.99, self.alpha)
    
    def exp2path(self, bandwidth_label: str):
        merge_suffix = ""
        if self.algorithm == "pipesd":
            merge_suffix = f"_merge={self.merge_policy}"
        tag_suffix = f"_tag={self.result_tag}" if self.result_tag else ""

        if 'edgeLLM' in self.exp_name:
            saved_path = os.path.join(self.exp_name, f"edgeLLM_alpha={self.args.init_alpha}_mult={self.multiply_times}{tag_suffix}_bw={bandwidth_label}MB.json")
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
    
    def edge_process_draft_model(self, prefix, task_id):
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
        self.max_len = prefix_len + self.max_generated_len
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

        if ('merge' in self.algorithm or 'pipesd' in self.algorithm) and not self.args.nomerge:
            merge_plan_batches = self._resolve_merge_plan()
            print(f"[Edge] 计算得到合并计划: {merge_plan_batches}")
        else:
            merge_plan_batches = [self.gamma * 4]

        merge_plan_index = 0
        
        total_start_time = time.time()
        while len(output_tokens) < self.max_len:
            
            # --- 1. 生成一个token ---
            next_token = self.draft_model.sample(top_k=self.top_k, top_p=self.top_p, temp=self.temp)
            current_probs = softmax(self.draft_model.scores[self.draft_model.n_tokens-1])
            self.num_spec_tokens_generated += 1
            self._spec_token_indices_generated.append(self.num_spec_tokens_generated)
            
            # 添加到全局推测序列
            total_speculative_tokens.append(next_token)
            total_speculative_indices.append(self._spec_token_indices_generated[-1])
            current_batch_tokens.append(next_token)
            current_batch_indices.append(self._spec_token_indices_generated[-1])
            
            start_eval = time.time()
            self.draft_model.eval([next_token])
            end_eval = time.time()
            eval_time = end_eval - start_eval
            if eval_time < self.args.default_token_compute:
                time.sleep(self.args.default_token_compute - eval_time)
            
            # 获取概率分布
            
            current_batch_probs.append(current_probs)
            total_speculative_probs.append(current_probs)

            max_probs.append(current_probs.max().item())
            
            if 'edgeLLM' in self.algorithm:
                self.verify_thresh_multi = self.alpha  # edgeLLM中多项式阈值等于alpha

            # --- 2. 检查发送和验证条件 ---
            should_send = (len(current_batch_tokens) >= merge_plan_batches[merge_plan_index])  # 时间条件

            should_verify = self.if_verify(
                total_speculative_probs,
                self.verify_strategy
            )

            should_end = (next_token == self.draft_model.token_eos())  # 结束条件

            # 如果同时满足发送和验证条件，优先验证（因为验证需要处理结果）
            if should_verify or should_end:
                merge_plan_index = 0  # 重置合并计划索引
                # 发起验证请求 - 使用当前的n_past值
                payload = {
                    'type': 'propose',
                    'tokens': current_batch_tokens.copy(),
                    'probs': [p.tolist() for p in current_batch_probs],
                    'task_id': task_id,
                    'n_past': current_n_past,  # 使用当前的n_past
                    'index': len(total_speculative_tokens) - len(current_batch_tokens),  # 本次验证的索引（从0开始）
                    'should_verify': True,  # 验证请求
                }
                payload_bytes = msgpack.packb(payload)
                # 将本轮所有推测token的索引计入“已验证”集合（避免重复计数）
                new_indices = [idx for idx in total_speculative_indices if idx not in self._spec_token_indices_sent]
                self._spec_token_indices_sent.update(new_indices)
                self.num_spec_tokens_sent += len(new_indices)

                # print(f"[DEBUG] 发送验证请求，tokens: {current_batch_tokens}, n_past: {current_n_past}， tokens: {self.draft_model.detokenize(total_speculative_tokens).decode('utf-8', 'ignore')}")

                future = self.sender.submit(PROPOSE_ENDPOINT, payload_bytes, headers={"Content-Type": "application/msgpack"})
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
                should_verify_waiting = False
                waiting_verify_future = None
                
                while self.send_while_generating and len(output_tokens) + len(total_speculative_tokens) + len(waiting_tokens) < self.max_len:

                    if not waiting_tokens:
                        speculated_final_token = self.draft_model.sample(top_k=self.top_k, top_p=self.top_p, temp=self.temp)
                        self.draft_model.eval([speculated_final_token])
                        self.num_spec_tokens_generated += 1
                        self._spec_token_indices_generated.append(self.num_spec_tokens_generated)
                    
                    wait_token = self.draft_model.sample(top_k=self.top_k, top_p=self.top_p, temp=self.temp)
                    wait_probs = softmax(self.draft_model.scores[self.draft_model.n_tokens-1])
                    self.num_spec_tokens_generated += 1
                    self._spec_token_indices_generated.append(self.num_spec_tokens_generated)
                    
                    # eval这个等待token
                    self.draft_model.eval([wait_token])
                    
                    waiting_tokens.append(wait_token)
                    waiting_probs.append(wait_probs)
                    waiting_indices.append(self._spec_token_indices_generated[-1])
                    
                    if 'edgeLLM' in self.algorithm:
                        self.verify_thresh_multi = self.alpha  # edgeLLM中多项式阈值等于alpha
                    
                    should_verify_waiting = self.if_verify(
                        waiting_probs,
                        self.verify_strategy
                    )
                    
                    if future.done():
                        verify_result = future.result()
                        if 'error' in verify_result or 'n_accepted' not in verify_result or 'final_token' not in verify_result:
                            print(f"[Edge] 服务器返回错误: {verify_result}")
                            return
                        n_accepted = verify_result['n_accepted']
                        final_token = verify_result['final_token']
                        if not (n_accepted == len(total_speculative_tokens) and final_token == speculated_final_token):
                            break

                    waiting_payload = {
                        'type': 'propose_waiting',
                        'tokens': [wait_token],
                        'probs': [wait_probs.tolist()],
                        'task_id': task_id,
                        'n_past': current_n_past + len(total_speculative_tokens) + 1,  # 更新n_past
                        'index': len(waiting_tokens) - 1,  # 更新索引
                        'should_verify': should_verify_waiting,  # 非验证请求
                    }
                    waiting_payload_bytes = msgpack.packb(waiting_payload)
                    if should_verify_waiting:
                        # 触发等待验证时，计入当前等待序列的索引
                        new_indices_wait = [idx for idx in waiting_indices if idx not in self._spec_token_indices_sent]
                        self._spec_token_indices_sent.update(new_indices_wait)
                        self.num_spec_tokens_sent += len(new_indices_wait)
                    
                    # 异步发送等待期间的批次
                    waiting_future = self.sender.submit(
                        PROPOSE_ENDPOINT,
                        waiting_payload_bytes,
                        headers={"Content-Type": "application/msgpack"},
                        tag=waiting_tag,
                    )
                    waiting_futures.append(waiting_future)
                    if should_verify_waiting:
                        waiting_verify_future = waiting_future
                    # print(f"[DEBUG] 发送等待期间的批次请求，tokens: {wait_token}, n_past: {current_n_past + len(total_speculative_tokens) + 1}， tokens: {self.draft_model.detokenize(waiting_tokens).decode('utf-8', 'ignore')}, should_verify: {should_verify_waiting}")

                    if should_verify_waiting or wait_token == self.draft_model.token_eos():
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
                self.verify_spec_lengths.append(len(total_speculative_tokens))
                self.verify_accept_lengths.append(n_accepted)
                self.verify_his.append((len(total_speculative_tokens), n_accepted))
                self.acc_ratio += n_accepted / len(total_speculative_tokens)
                
                # 更新输出tokens
                accepted_tokens = total_speculative_tokens[:n_accepted]
                output_tokens.extend(accepted_tokens)
                output_tokens.append(final_token)
                self._record_token_time(len(accepted_tokens) + 1)
                
                last_verify_all_passed = (n_accepted == len(total_speculative_tokens) and final_token == speculated_final_token)
                if not last_verify_all_passed:
                    self.draft_model.n_tokens = current_n_past + n_accepted
                # print(f'当前n_tokens: {self.draft_model.n_tokens}, current_n_past: {current_n_past}, n_accepted: {n_accepted}')
                current_n_past = current_n_past + n_accepted + 1
                if final_token != speculated_final_token:
                    self.draft_model.eval([final_token])
                    # print(f"[DEBUG] final_token 与 speculated_final_token 不同，eval final_token: {final_token}")
                
                # print(f"[DEBUG] 更新状态: n_past {current_n_past - n_accepted - 1} -> {current_n_past}, accepted {n_accepted}, final_token {final_token}")
                
                if self.verify_strategy == 'multiple-tokens':
                    # 更新多项式阈值
                    self.update_thresh(multiply_times=self.multiply_times, n_accepted=n_accepted, n_all=len(total_speculative_tokens))
                    # print(f"[DEBUG] 更新多项式阈值: verify_thresh_multi={self.verify_thresh_multi:.6f}, accumulated_probs={self.accumulated_probs:.6f}")
                
                verify_result = None  # 重置验证结果
                
                
                # --- 5. 处理等待期间生成的token ---
                # print(f"[DEBUG] 验证期间生成了 {len(waiting_tokens)} 个token，n_accepted={n_accepted}")
                waiting_batch_future = None
                if waiting_tokens:
                    # print(f"[DEBUG] final_token: {final_token}, speculated_final_token: {speculated_final_token}, n_accepted: {n_accepted}, total_speculative_tokens: {len(total_speculative_tokens)}")
                    # 仅重打包那些尚未出队发送的等待请求：排除已完成/已取消/已 in-flight 的 future
                    pending_indices = []
                    for idx, fut in enumerate(waiting_futures):
                        if fut.done() or fut.cancelled():
                            continue
                        try:
                            inflight = self.sender.is_inflight_future(fut)
                        except Exception:
                            inflight = False
                        if inflight:
                            continue
                        pending_indices.append(idx)
                    if last_verify_all_passed:
                        if should_verify_waiting:
                            if pending_indices:
                                for idx in pending_indices:
                                    fut = waiting_futures[idx]
                                    if not fut.done():
                                        self.sender.cancel_future(fut)
                                drained_requests = self.sender.drain_tag(waiting_tag)
                                # if drained_requests:
                                    # print(f"[DEBUG] 取消等待期间挂起的 {len(drained_requests)} 个请求以重新打包")
                                waiting_batch_tokens = [waiting_tokens[idx] for idx in pending_indices]
                                waiting_batch_probs = [waiting_probs[idx] for idx in pending_indices]
                                waiting_batch_payload = {
                                    'type': 'propose_waiting',
                                    'tokens': waiting_batch_tokens.copy(),
                                    'probs': [p.tolist() for p in waiting_batch_probs],
                                    'task_id': task_id,
                                    'n_past': current_n_past,
                                    'index': len(waiting_tokens) - len(waiting_batch_tokens),
                                    'should_verify': True,
                                }
                                waiting_batch_bytes = msgpack.packb(waiting_batch_payload)
                                waiting_batch_indices = [waiting_indices[idx] for idx in pending_indices]
                                new_indices_wait_batch = [idx for idx in waiting_batch_indices if idx not in self._spec_token_indices_sent]
                                self._spec_token_indices_sent.update(new_indices_wait_batch)
                                self.num_spec_tokens_sent += len(new_indices_wait_batch)
                                waiting_batch_future = self.sender.submit(
                                    PROPOSE_ENDPOINT,
                                    waiting_batch_bytes,
                                    headers={"Content-Type": "application/msgpack"},
                                    tag=f"{waiting_tag}_batch",
                                )
                                waiting_futures = [waiting_futures[idx] for idx in range(len(waiting_futures)) if idx not in pending_indices]
                                waiting_futures.append(waiting_batch_future)
                            else:
                                waiting_batch_future = waiting_verify_future
                                # print("[DEBUG] 等待请求已全部完成，跳过重新打包")
                        else:
                            total_speculative_tokens = waiting_tokens
                            total_speculative_probs = waiting_probs
                            total_speculative_indices = waiting_indices
                            current_batch_tokens = []
                            current_batch_probs = []
                            current_batch_indices = []
                            # print(f"[DEBUG] 全部接受，使用等待期间的 {len(waiting_tokens)} 个token作为新推测")
                    else:
                        for fut in waiting_futures:
                            if not fut.done():
                                self.sender.cancel_future(fut)
                        drained_requests = self.sender.drain_tag(waiting_tag)
                        # if drained_requests:
                        #     print(f"[DEBUG] 验证未通过，取消等待期间的 {len(drained_requests)} 个挂起请求")

                    if waiting_batch_future is not None and should_verify_waiting:
                        verify_result_waiting = waiting_batch_future.result()
                        if 'n_accepted' not in verify_result_waiting:
                            print(f"[Edge] 服务器返回错误: {verify_result_waiting}")
                            return
                        n_accepted_waiting = verify_result_waiting['n_accepted']
                        final_token_waiting = verify_result_waiting['final_token']
                        waiting_spec_len = self._resolve_waiting_verify_length(
                            waiting_tokens=waiting_tokens,
                            waiting_batch_tokens=waiting_batch_tokens,
                        )
                        self.verify_spec_lengths.append(waiting_spec_len)
                        self.verify_accept_lengths.append(n_accepted_waiting)
                        self.verify_his.append((waiting_spec_len, n_accepted_waiting))
                        if waiting_spec_len > 0:
                            self.acc_ratio += n_accepted_waiting / waiting_spec_len
                        accepted_waiting_tokens = waiting_tokens[:n_accepted_waiting] if waiting_tokens else []
                        output_tokens.extend(accepted_waiting_tokens)
                        output_tokens.append(final_token_waiting)
                        self._record_token_time(len(accepted_waiting_tokens) + 1)
                        self.draft_model.n_tokens = current_n_past + n_accepted_waiting
                        # print(f'当前n_tokens: {self.draft_model.n_tokens}, current_n_past: {current_n_past}, n_accepted_waiting: {n_accepted_waiting}')
                        self.draft_model.eval([final_token_waiting])
                        current_n_past = self.draft_model.n_tokens
                        # print(f"[DEBUG] 重构批次全部接受，更新状态: n_past {current_n_past - n_accepted_waiting - 1} -> {current_n_past}, final_token {final_token_waiting}")
                        if self.verify_strategy == 'multiple-tokens':
                            # 更新多项式阈值
                            self.update_thresh(multiply_times=self.multiply_times, n_accepted=n_accepted_waiting, n_all=len(waiting_tokens))
                            # print(f"[DEBUG] 更新多项式阈值: verify_thresh_multi={self.verify_thresh_multi:.6f}, accumulated_probs={self.accumulated_probs:.6f}")

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
                }
                payload_bytes = msgpack.packb(payload)

                # print(f"[DEBUG] 发送批次请求，tokens: {current_batch_tokens}, n_past: {current_n_past}, index: {len(total_speculative_tokens)}， tokens: {self.draft_model.detokenize(total_speculative_tokens).decode('utf-8', 'ignore')}")
                # print(f"[发送] 当前n_tokens: {self.draft_model.n_tokens}, current_n_past: {current_n_past}")

                future = self.sender.submit(PROPOSE_ENDPOINT, payload_bytes, headers={"Content-Type": "application/msgpack"})

                # 重置当前批次
                current_batch_tokens = []
                current_batch_probs = []
                current_batch_indices = []
                merge_plan_index = min(merge_plan_index + 1, len(merge_plan_batches) - 1)

        total_end_time = time.time()
        spent_time = total_end_time - total_start_time
        self.color_print(f"[Edge] 任务 {task_id} 处理完成，输出长度 {len(output_tokens) - prefix_len}，总耗时: {spent_time:.4f} 秒, 单位token耗时 {spent_time / (len(output_tokens) - prefix_len):.4f} 秒", 5)

        decoded_text = self.draft_model.detokenize(output_tokens).decode("utf-8", "ignore")
        post_result = self.postprocess(prefix, decoded_text)
        # self.color_print(f"[Edge] 任务 {task_id} 生成完成, 前缀长度{prefix_len}，输出长度 {len(output_tokens) - prefix_len}，结果:\n{post_result}", 2)
        verify_stats = {
            'num_verifications': len(self.verify_spec_lengths),
            'num_spec_tokens_sent': self.num_spec_tokens_sent,
            'num_spec_tokens_generated': self.num_spec_tokens_generated,
            'num_spec_tokens': self.num_spec_tokens_sent,
        }

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
        avg_token_time = float(sum(self._token_durations) / len(self._token_durations)) if self._token_durations else None
        diagnostics = self._build_verify_diagnostics(output_length)
        exp_result = {
            'task_id': task_id,
            'output_length': output_length,
            # 'counted_length': eff_num,
            'total_time': spent_time,
            'output': decoded_text,
            'gamma': self.gamma,
            'max_len': self.max_len,
            'strategy': self.verify_strategy,
            'merge_policy': self.merge_policy,
            'bandwidth_MBps': self.bandwidth_MBps,
            'thresh_single': self.verify_thresh_single,
            'thresh_multi': self.verify_thresh_multi,
            'verify_stats': verify_stats,
            'token_durations': self._token_durations,
            'avg_token_time': avg_token_time,
            'gpu_power_integral_joules': exit_result.get('gpu_power_integral_joules', None),
            'verify_num': exit_result.get('verify_num', None),
            'acc_ratio': self.acc_ratio / len(self.verify_spec_lengths) if self.verify_spec_lengths else 0.0,
            'verify_spec_lengths': self.verify_spec_lengths,
            'verify_accept_lengths': self.verify_accept_lengths,
            'verify_his': self.verify_his,
            'diagnostics': diagnostics,
        }
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
        data.append(exp_result)

        # 写回整个文件
        with open(saved_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        self.sender.close()

        return post_result, spent_time

        

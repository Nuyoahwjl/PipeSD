import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np

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
        "codellama-7b": 32000,
        "codellama-34b": 32000,
        "codellama-70b": 32000,
        "TinyLlama-1.1B-Chat-v1.0-GPTQ": 32000,
        "tinyllama-1.1b-chat-v1.0-gguf": 32000,
        "llama-2-70b": 32000,
        # "deepseek-1.3b": 32256,
        # "deepseek-6.7b": 32256,
        "deepseek-1.3b": 32256,
        "deepseek-coder-1.3b-base-GGUF": 32256,
        "deepseek-coder-1.3b-instruct-GGUF": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-coder-6.7B-instruct-GGUF": 32256,
        "deepseek-33b": 32256,
    }
    
    zoo = {
        "codellama-7b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "codellama-34b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "codellama-70b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "tinyllama-1.1b-chat-v1.0-gguf": "pre_models/tinyllama-1.1b-chat-v1.0-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        # "tinyllama-1.1b-chat-v1.0-gguf": "pre_models/qwen1.5/qwen1_5-1_8b-chat-q4_k_m.gguf",
        "llama-2-7b": "pre_models/llama-2-7b-chat-gguf/llama-2-7b-chat.Q4_K_M.gguf",
        "llama-2-70b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        # "deepseek-1.3b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "deepseek-6.7b": "pre_models/deepseek-coder-6.7B-instruct-GPTQ",
        # "deepseek-6.7b": "/home/jianhongbai/gyq/FastSD/deepseek-1.3b",
        # "deepseek-33b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        # "deepseek-1.3b": "/home/jianhongbai/gyq/FastSD/deepseek-coder-1.3b-gptq",
        "deepseek-1.3b": "pre_models/deepseek-coder-1.3b-instruct-GPTQ",
        "deepseek-coder-1.3b-base-GGUF": "/home/jianhongbai/gyq/FastSD/deepseek-coder-1.3b-base-GGUF",
        "deepseek-coder-1.3b-instruct-GGUF": "pre_models/deepseek-coder-1.3b-instruct-GGUF/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
        "deepseek-coder-6.7B-instruct-GGUF": "pre_models/deepseek-coder-6.7B-instruct-GGUF/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "deepseek-coder-33B-instruct-GGUF": "pre_models/deepseek-coder-33B-instruct-GGUF/deepseek-coder-33b-instruct.Q4_K_M.gguf",
        "deepseek-33b": "deepseek-ai/deepseek-coder-33b-base",
    }

    # args.vocab_size = vocab_size[args.draft_model]
    args.draft_model = zoo[args.draft_model]
    # args.target_model = zoo[args.target_model]

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    # parser.add_argument('--dataset', type=str, default="humaneval")
    # parser.add_argument('--data_path', type=str, default="data/humaneval.jsonl")
    # parser.add_argument('--dataset', type=str, default="mt_bench")
    parser.add_argument('--dataset', type=str, default="gsm8k")
    # parser.add_argument('--data_path', type=str, default="data/mt_bench.jsonl")

    # gsm8k data path
    # parser.add_argument('--data_path', type=str, default="/home/jianhongbai/gyq/FastSD/ParallelSpeculativeDecoding-main/data/gsm8k.jsonl")
    # humaneval data path
    
    # mt_bench data path
    # parser.add_argument('--data_path', type=str, default="/home/jianhongbai/gyq/FastSD/ParallelSpeculativeDecoding-main/data/mt_bench.jsonl")

    # parser.add_argument('--draft_model', type=str, default="deepseek-coder-1.3b-instruct-GGUF")
    # parser.add_argument('--target_model', type=str, default="deepseek-coder-6.7B-instruct-GGUF")
    parser.add_argument('--draft_model', type=str, default="tinyllama-1.1b-chat-v1.0-gguf")
    parser.add_argument('--target_model', type=str, default="deepseek-coder-6.7B-instruct-GGUF")
    
    parser.add_argument('--exp_name', '-e', type=str, default="exp_fixednum", help='folder name for storing results.')
    parser.add_argument('--eval_mode', type=str, default="small", choices=["small", "large", "sd", "para_sd", "para_sd_wo_1", "para_sd_wo_2"], help='eval mode.')
    parser.add_argument('--num_samples_per_task', '-n', type=int, default=1, help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=26, help='max token number generated.')
    parser.add_argument('--max_generated_tokens', type=int, default=128, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=1, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=6, help='guess time.')
    parser.add_argument("--num_drafts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--token_budget", type=int, default=400)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ctx_size", type=int, default=16384)
    parser.add_argument("--verify_thresh_single", type=float, default=0.94)
    parser.add_argument("--verify_thresh_multi", type=float, default=0.9)
    parser.add_argument("--verify_thresh_diff", type=float, default=0.05)
    parser.add_argument("--accumulated_num", type=int, default=5)
    # parser.add_argument("--verify_mode", type=str, default="hybrid", choices=["single", "multi", "hybrid"])
    parser.add_argument('--excluded_num', type=int, default=100, help='excluded num of the beginning.')
    parser.add_argument('--C', type=float, default=0.05, help='startup cost')
    parser.add_argument('--verify_strategy', type=str, default="fixed-num", choices=["fixed-num", "single-token", "multiple-tokens", "hybrid", "pid", "diff", "entropy"], help='verification strategy.')
    parser.add_argument('--verify_num', type=int, default=8, help='number of tokens to verify in fixed-num strategy.')
    parser.add_argument('--bayes_optimize', action='store_true', help='Enable Bayesian optimization over hybrid thresholds.')
    parser.add_argument('--bayes_calls', type=int, default=15, help='Total Bayesian optimization iterations.')
    parser.add_argument('--bayes_init_points', type=int, default=5, help='Number of random init points before GP proposals.')
    parser.add_argument('--bayes_single_min', type=float, default=0.6, help='Lower bound for verify_thresh_single search.')
    parser.add_argument('--bayes_single_max', type=float, default=0.99, help='Upper bound for verify_thresh_single search.')
    parser.add_argument('--bayes_multi_min', type=float, default=0.05, help='Lower bound for verify_thresh_multi search.')
    parser.add_argument('--bayes_multi_max', type=float, default=0.9, help='Upper bound for verify_thresh_multi search.')
    parser.add_argument('--bayes_tokens_per_trial', type=int, default=50, help='Minimum tokens aggregated per trial for latency averaging.')
    parser.add_argument('--init_alpha', type=float, default=0.92, help='Initial alpha value for some parameter.')
    parser.add_argument('--multiply_times', type=float, default=0.95, help='Decay rate for alpha parameter.')
    parser.add_argument('--pid_target_accept', type=float, default=0.75, help='Target acceptance rate for PID verify strategy.')
    parser.add_argument('--pid_kp', type=float, default=0.1, help='Proportional gain for PID verify strategy.')
    parser.add_argument('--pid_ki', type=float, default=0.01, help='Integral gain for PID verify strategy.')
    parser.add_argument('--pid_kd', type=float, default=0.02, help='Derivative gain for PID verify strategy.')
    parser.add_argument('--pid_init_tau', type=float, default=0.1, help='Initial logit-gap threshold for PID verify strategy.')
    parser.add_argument('--pid_tau_min', type=float, default=0.01, help='Minimum allowed logit-gap threshold.')
    parser.add_argument('--pid_tau_max', type=float, default=0.7, help='Maximum allowed logit-gap threshold.')
    parser.add_argument('--entropy_thresh', type=float, default=0.1, help='Entropy threshold for entropy verify strategy.')
    # Bandwidth limiting / networking related args
    parser.add_argument('--bandwidth_limit_kbps', type=float, default=2048, help='Outbound bandwidth limit in KB/s (default 2048 KB/s = 2 MB/s). Set to 0 or negative to disable.')
    parser.add_argument('--bucket_size_bytes', type=int, default=None, help='Token bucket capacity in bytes. If not set, defaults to rate_bps * 1 second (allows 1s burst).')
    parser.add_argument('--init_rtt', type=float, default=0.05, help='Initial RTT estimate in seconds used by the bandwidth limiter.')
    # parser.add_argument('--rtt_alpha', type=float, default=0.2, help='EWMA alpha for RTT smoothing (0..1).')
    parser.add_argument('--safety_margin', type=float, default=0.05, help='Safety margin multiplier applied to computed communication time (e.g., 0.05 = +5%%).')
    parser.add_argument('--min_sleep_ms', type=int, default=10, help='Minimum sleep (ms) enforced to avoid busy-waiting for tiny waits.')
    parser.add_argument('--bandwidth_debug', action='store_true', default=False, help='Enable debug logging for the bandwidth limiter.')
    parser.add_argument('--trace_speculative', action='store_true', default=False, help='Enable speculative decoding trace logging.')
    parser.add_argument('--speculative_trace_path', type=str, default=None, help='Path to save speculative trace logs.')
    parser.add_argument('--bandwidth_MBps', type=float, default=2.5, help='bandwidth limit in MB/s.')
    parser.add_argument('--baseline_test', action='store_true', help='Use full-merge baseline (send accumulated tokens only at verification).')
    parser.add_argument('--edgeLLM', action='store_true', help='Use full-merge baseline (send accumulated tokens only at verification).')
    parser.add_argument('--merge_plan_interval', type=int, default=100, help='Number of generated tokens between merge plan recomputations.')
    parser.add_argument('--default_token_compute', type=float, default=0.144, help='Default single-token compute time used for planning.')
    parser.add_argument('--token_size_MB', type=float, default=0.29, help='Average token size in MB used for planning.')
    # parser.add_argument('--send_while_generating', action='store_true', help='Enable sending tokens while generating to overlap communication and computation.')
    parser.add_argument('--algorithm', type=str, default="vanilla", choices=["vanilla", "vanilla-with-send", 'vanilla-with-merge', 'vanilla-with-merge-no-send', 'edgeLLM', 'hsl', 'pipesd'], help='Description of some other argument.')
    parser.add_argument('--start_index_of_sample', type=int, default=0, help='start index of samples to eval.')
    parser.add_argument('--end_index_of_sample', type=int, default=4, help='end index of samples to eval.')
    parser.add_argument('--ablation_study', action='store_true', help='whether to run ablation study.')
    parser.add_argument('--nomerge', action='store_true', help='whether to run no merge version.')
    args = parser.parse_args() 
    args = args_proc(args)
    model_zoo(args)
    # print(args)
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
        # args.target_model = "llama-2-7b"
        # args.draft_model = "llama-2-7b"
        # args.target_model = "llama-2-7b"

    if args.algorithm == 'edgeLLM':
            args.verify_strategy = 'multiple-tokens'
    elif args.algorithm == 'hsl':
            args.verify_strategy = 'single-token'
    elif args.algorithm == 'pipesd':
            print(args.ablation_study, args.nomerge)
            if (not args.ablation_study) or args.nomerge:
                args.verify_strategy = 'hybrid'
    return args

def strategy2exp(strategy: str):
    if strategy == 'fixed-num':
        return 'exp_fixednum'
    elif strategy == 'single-token':
        return 'exp_single_norestrict'
    elif strategy == 'multiple-tokens':
        return 'exp_edgeLLM_norestrict'
    elif strategy == 'hybrid':
        return 'exp_hybrid_norestrict'
    elif strategy == 'diff':
        return 'exp_diff'
    elif strategy == 'entropy':
        return 'exp_entropy'

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    if temperature == 0:
        idx = logits.argmax(dim=1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits[:, idx] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def sample(probs, num_samples: int = 1):
    """
    从概率分布 probs 中采样 num_samples 个索引
    支持 torch.Tensor 和 numpy.ndarray
    """
    # Torch 情况
    if isinstance(probs, torch.Tensor):
        # 确保是 1D 概率分布
        if probs.dim() > 1:
            raise ValueError("只支持 1D 概率分布 (shape = [vocab_size])")
        probs = probs / probs.sum()  # 归一化
        idx_next = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        return idx_next.cpu().numpy()  # 统一返回 numpy

    elif isinstance(probs, np.ndarray):
        if probs.ndim > 1:
            raise ValueError("只支持 1D 概率分布 (shape = [vocab_size])")
        # probs = probs / probs.sum()
        # rng = np.random.default_rng(seed)
        # idx_next = np.random.choice(len(probs), size=num_samples, p=probs)
        # return int(idx_next[0])
        probs = probs / probs.sum()
        return int(np.argmax(probs))

    else:
        raise TypeError("probs 必须是 torch.Tensor 或 numpy.ndarray")

def max_fn(x):
    """
    统一支持PyTorch和NumPy的正数归一化函数
        norm(max (x, 0))
    
    参数:
        x: torch.Tensor 或 numpy.ndarray

    返回:
        与输入类型相同的归一化结果
    """
    
    # 判断输入类型
    if isinstance(x, torch.Tensor):
        # PyTorch版本
        x_max = torch.where(x > 0, x, torch.zeros_like(x))
        x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
        # 避免除零错误
        x_max_sum = torch.where(x_max_sum > 0, x_max_sum, torch.ones_like(x_max_sum))
        return x_max / x_max_sum
    
    elif isinstance(x, np.ndarray):
        if x.ndim not in [1, 2]:
             raise ValueError(f"NumPy input must be 1D or 2D, got {x.ndim}D")
        # NumPy: max(0, x)
        x_max = np.maximum(x, 0.0) # 更简洁的写法
        # 归一化: 对最后一个维度 (词汇表维度) 求和
        x_max_sum = np.sum(x_max, axis=-1, keepdims=True) # axis=-1 适应 1D/2D
        # 防止除零
        x_max_sum = np.clip(x_max_sum, a_min=1e-12, a_max=None) # 或者使用 np.where
        return x_max / x_max_sum

    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # 防止溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

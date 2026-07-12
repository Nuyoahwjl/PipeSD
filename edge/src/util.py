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

def model_zoo(args):
    vocab_size = {
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
        "tinyllama-1.1b-chat-v1.0-gguf": "pre_models/tinyllama-1.1b-chat-v1.0-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "deepseek-coder-1.3b-instruct-GGUF": "pre_models/deepseek-coder-1.3b-instruct-GGUF/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
    }

    # args.vocab_size = vocab_size[args.draft_model]
    args.draft_model = zoo[args.draft_model]
    # args.target_model = zoo[args.target_model]

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--dataset', type=str, default="humaneval")
    # parser.add_argument('--dataset', type=str, default="gsm8k")
    parser.add_argument('--exp_name', '-e', type=str, default="exp_fixednum", help='folder name for storing results.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_generated_tokens', type=int, default=1024, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=1, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=6, help='guess time.')
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ctx_size", type=int, default=16384)
    parser.add_argument("--draft_n_gpu_layers", type=int, default=0, help="Number of GPU layers used by the local draft model.")
    parser.add_argument("--verify_thresh_single", type=float, default=0.94)
    parser.add_argument("--verify_thresh_multi", type=float, default=0.9)
    parser.add_argument('--C', type=float, default=0.025, help='startup cost')
    parser.add_argument('--verify_strategy', type=str, default="fixed-num", choices=["fixed-num", "single-token", "multiple-tokens", "hybrid"], help='verification strategy.')
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
    parser.add_argument('--init_rtt', type=float, default=0.05, help='Initial RTT estimate in seconds used by the bandwidth limiter.')
    parser.add_argument('--bandwidth_MBps', type=float, default=2.5, help='bandwidth limit in MB/s.')
    parser.add_argument('--server_timeout_s', type=int, default=10, help='HTTP timeout in seconds for cloud requests.')
    parser.add_argument('--baseline_test', action='store_true', help='Use full-merge baseline (send accumulated tokens only at verification).')
    parser.add_argument('--edgeLLM', action='store_true', help='Use full-merge baseline (send accumulated tokens only at verification).')
    parser.add_argument('--default_token_compute', type=float, default=0.036, help='Default single-token compute time used for planning.')
    parser.add_argument('--token_size_MB', type=float, default=0.29, help='Average token size in MB used for planning.')
    parser.add_argument(
        '--merge_policy',
        type=str,
        default='dp',
        choices=['dp', 'immediate', 'no_early'],
        help='Merge scheduling policy used by pipesd during speculative upload.',
    )
    parser.add_argument('--result_tag', type=str, default="", help='Optional tag appended to result filenames to isolate experiment runs.')
    parser.add_argument('--task_id_offset', type=int, default=0, help='Offset added to each task id to avoid collisions across concurrent clients.')
    parser.add_argument(
        '--use_env_proxy',
        action='store_true',
        help='Respect HTTP(S)_PROXY and related environment variables for network communication.',
    )
    # parser.add_argument('--send_while_generating', action='store_true', help='Enable sending tokens while generating to overlap communication and computation.')
    parser.add_argument('--algorithm', type=str, default="vanilla", choices=["vanilla", 'edgeLLM', 'hsl', 'pipesd'], help='Different algorithm choices.')
    parser.add_argument('--start_index_of_sample', type=int, default=0, help='start index of samples to eval.')
    parser.add_argument('--end_index_of_sample', type=int, default=0, help='end index of samples to eval.')
    parser.add_argument('--max_samples', type=int, default=None, help='optional cap on the number of loaded samples, useful for quick debugging.')
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

    if args.algorithm == 'edgeLLM':
        args.verify_strategy = 'multiple-tokens'
    elif args.algorithm == 'hsl':
        args.verify_strategy = 'single-token'
    elif args.algorithm == 'pipesd':
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

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # 防止溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

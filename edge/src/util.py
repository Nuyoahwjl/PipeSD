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

    args.vocab_size = vocab_size[args.draft_model]
    args.draft_model = zoo[args.draft_model]
    # args.target_model = zoo[args.target_model]

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--dataset', type=str, default="humaneval")
    # parser.add_argument('--dataset', type=str, default="gsm8k")
    parser.add_argument('--exp_name', '-e', type=str, default="exp_fixednum", help='folder name for storing results.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_generated_tokens', type=int, default=128, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=1, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument(
        '--prob_transport',
        choices=['full', 'lazy_distribution'],
        default='full',
        help=(
            'full preserves the original protocol; lazy_distribution sends only '
            'q(draft_token) and uploads one full distribution after a rejection.'
        ),
    )
    parser.add_argument('--gamma', type=int, default=6, help='guess time.')
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ctx_size", type=int, default=16384)
    parser.add_argument("--draft_n_gpu_layers", type=int, default=0, help="Number of GPU layers used by the local draft model.")
    parser.add_argument("--verify_thresh_single", type=float, default=0.94)
    parser.add_argument("--verify_thresh_multi", type=float, default=0.9)
    parser.add_argument('--C', type=float, default=0.025, help='startup cost')
    parser.add_argument('--verify_strategy', type=str, default="fixed-num", choices=["fixed-num", "single-token", "multiple-tokens", "hybrid"], help='verification strategy.')
    parser.add_argument('--verify_num', type=int, default=8, help='number of tokens to verify in fixed-num strategy.')
    parser.add_argument('--bayes_optimize', action='store_true', help='Enable PipeSD Bayesian optimization over hybrid thresholds.')
    parser.add_argument('--bayes_only', action='store_true', help='Run BO and exit without formal evaluation.')
    parser.add_argument('--bayes_calls', type=int, default=16, help='Total BO samples (16 in the paper).')
    parser.add_argument('--bayes_init_points', type=int, default=1, help='Random initial samples (one in the paper).')
    parser.add_argument('--bayes_single_min', type=float, default=1e-6, help='Lower bound for R2 search.')
    parser.add_argument('--bayes_single_max', type=float, default=1.0, help='Upper bound for R2 search.')
    parser.add_argument('--bayes_multi_min', type=float, default=1e-6, help='Lower bound for R1 search.')
    parser.add_argument('--bayes_multi_max', type=float, default=1.0, help='Upper bound for R1 search.')
    parser.add_argument(
        '--bo_protocol',
        choices=['paper', 'sample_coverage'],
        default='paper',
        help='paper: 20 accepted tokens total per candidate; sample_coverage: a per-sample budget.',
    )
    parser.add_argument('--bayes_tokens_per_sample', type=int, default=None, help='Accepted tokens per selected sample in sample_coverage BO mode.')
    parser.add_argument('--bayes_tokens_per_trial', type=int, default=20, help='Accepted tokens total per candidate in paper BO mode.')
    parser.add_argument('--bayes_ei_xi', type=float, default=0.1, help='Expected-improvement exploration parameter.')
    parser.add_argument('--bo_config_path', type=str, default='', help='BO configuration used by a formal evaluation; recorded for provenance.')
    parser.add_argument('--init_alpha', type=float, default=0.92, help='Initial EdgeLLM cumulative-confidence threshold R1.')
    parser.add_argument('--multiply_times', type=float, default=0.95, help='Decay rate for alpha parameter.')
    parser.add_argument('--edge_llm_full_accept_decay', type=float, default=0.5, help='Paper Eq. (7) decay applied after a fully accepted EdgeLLM round.')
    parser.add_argument('--init_rtt', type=float, default=0.05, help='Deprecated compatibility field; software timing uses the explicit startup parameters below.')
    parser.add_argument('--bandwidth_MBps', '--uplink_bandwidth_MBps', dest='bandwidth_MBps', type=float, default=2.5, help='Edge-to-cloud uplink limit in MB/s (2.5 MB/s in Scenario 1).')
    parser.add_argument('--downlink_bandwidth_MBps', type=float, default=25.0, help='Cloud-to-edge downlink limit in MB/s (25 MB/s in Scenario 1). Software mode enforces it in the shared link emulator.')
    parser.add_argument(
        '--network_shaping_mode',
        choices=['software', 'os'],
        default='software',
        help='software uses a shared pre-delivery uplink/downlink emulator; os relies on tc/QoS and measures real transport time.',
    )
    parser.add_argument(
        '--software_uplink_startup_ms',
        type=float,
        default=None,
        help='Fixed per-upload startup alpha in software mode. Defaults to --C converted to milliseconds.',
    )
    parser.add_argument(
        '--software_downlink_startup_ms',
        type=float,
        default=0.0,
        help='Fixed per-response startup in software mode. Keep 0 for the paper Scenario 1 unless calibrating a measured RTT.',
    )
    parser.add_argument(
        '--software_bandwidth_profile',
        type=str,
        default='',
        help='Optional repeating Scenario-4 profile as up_MBps:down_MBps pairs separated by commas.',
    )
    parser.add_argument(
        '--software_bandwidth_change_interval_s',
        type=float,
        default=20.0,
        help='Seconds between software bandwidth-profile entries (20 in paper Scenario 4).',
    )
    parser.add_argument('--server_timeout_s', type=int, default=10, help='HTTP timeout in seconds for cloud requests.')
    parser.add_argument('--baseline_test', action='store_true', help='Use full-merge baseline (send accumulated tokens only at verification).')
    parser.add_argument('--edgeLLM', action='store_true', help='Use full-merge baseline (send accumulated tokens only at verification).')
    parser.add_argument('--initial_generation_gamma', type=float, default=0.036, help='Initial per-token generation estimate used only by the DP planner.')
    parser.add_argument('--default_token_compute', type=float, default=None, help='Deprecated alias for --initial_generation_gamma; it no longer enables artificial delay.')
    parser.add_argument('--enable_compute_emulation', action='store_true', help='Enable artificial per-token delay for emulated Scenario 2/3 only.')
    parser.add_argument('--emulated_generation_delay', type=float, default=0.0, help='Extra seconds slept after each generated token when compute emulation is enabled.')
    parser.add_argument('--token_size_MB', type=float, default=0.29, help='Average token size in MB used for planning.')
    parser.add_argument('--schedule_window', type=int, default=20, help='Initial DP scheduling window N-hat.')
    parser.add_argument('--schedule_history_size', type=int, default=100, help='Draft sequences used for moving-average N-hat.')
    parser.add_argument('--environment_update_threshold', type=float, default=None, help='Deprecated shared threshold for all online updates.')
    parser.add_argument('--tpt_update_threshold', type=float, default=0.2, help='TPT relative-change threshold delta1.')
    parser.add_argument('--gamma_update_threshold', type=float, default=0.2, help='Generation-time relative-change threshold delta2.')
    parser.add_argument('--communication_update_threshold', type=float, default=0.2, help='Communication alpha/beta relative-change threshold delta3.')
    parser.add_argument('--regression_min_comm_samples', type=int, default=8, help='Minimum communication samples before alpha/beta regression.')
    parser.add_argument(
        '--lazy_comm_probe_sizes',
        type=str,
        default='1,4,16,64,256,1024,2048,4096',
        help='Comma-separated scalar-probability probe sizes used only by lazy_distribution.',
    )
    parser.add_argument(
        '--lazy_comm_probe_repetitions',
        type=int,
        default=3,
        help='Measurements per lazy communication probe size; medians reduce HTTP jitter.',
    )
    parser.add_argument(
        '--lazy_comm_min_r_squared',
        type=float,
        default=0.8,
        help='Minimum payload-byte regression R2 required before lazy alpha/beta update.',
    )
    parser.add_argument(
        '--disable_online_environment_measurement',
        action='store_true',
        help='Disable online alpha/beta/gamma measurement updates for DP scheduling.',
    )
    parser.add_argument(
        '--merge_policy',
        type=str,
        default='dp',
        choices=['dp', 'immediate', 'no_early'],
        help='Merge scheduling policy used by pipesd during speculative upload.',
    )
    parser.add_argument('--result_tag', type=str, default="", help='Optional tag appended to result filenames to isolate experiment runs.')
    parser.add_argument(
        '--evaluation_protocol',
        choices=['sample_index', 'paper_table1'],
        default='sample_index',
        help='sample_index preserves the legacy index-based debug run; paper_table1 stops at a shared accepted-token budget.',
    )
    parser.add_argument('--target_output_tokens', type=int, default=1000, help='Accepted-token budget for paper_table1.')
    parser.add_argument('--run_id', type=str, default='', help='Stable run identifier; generated automatically when omitted.')
    parser.add_argument('--task_id_offset', type=int, default=0, help='Offset added to each task id to avoid collisions across concurrent clients.')
    parser.add_argument('--client_id', type=int, default=0, help='Logical edge client id recorded in multi-client results.')
    parser.add_argument('--run_duration_s', type=float, default=0.0, help='Measured closed-loop workload duration; 0 keeps the index-based one-pass behavior.')
    parser.add_argument('--warmup_duration_s', type=float, default=0.0, help='Warm-up duration before the measured window.')
    parser.add_argument('--barrier_dir', type=str, default='', help='Directory used by the launcher to synchronize model-ready clients.')
    parser.add_argument('--barrier_timeout_s', type=float, default=1800.0)
    parser.add_argument('--workload_seed', type=int, default=3407)
    parser.add_argument('--software_bandwidth_profile_offset', type=int, default=0, help='Rotate the dynamic link trace per edge to avoid synchronized synthetic links.')
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
    if getattr(args, 'default_token_compute', None) is not None:
        args.initial_generation_gamma = args.default_token_compute
    if getattr(args, 'environment_update_threshold', None) is not None:
        args.tpt_update_threshold = args.environment_update_threshold
        args.gamma_update_threshold = args.environment_update_threshold
        args.communication_update_threshold = args.environment_update_threshold

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

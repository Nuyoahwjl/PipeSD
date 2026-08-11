#!/usr/bin/env python3
"""Run co-located pure-edge or pure-cloud speculative decoding."""

import argparse
import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))

from src.local_speculative import LocalSpeculativeEvaluator


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Co-located fixed-window speculative decoding. Both the draft and "
            "target model run on the selected pure_edge or pure_cloud endpoint."
        )
    )
    parser.add_argument("--mode", required=True, choices=("pure_edge", "pure_cloud"))
    parser.add_argument("--dataset", required=True, choices=("humaneval", "gsm8k"))
    parser.add_argument("--draft_model_path", default="")
    parser.add_argument("--target_model_path", default="")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--output_root", default="exp/exp__wjl")
    parser.add_argument("--result_tag", default="four_mode_s1_paper")
    parser.add_argument("--run_id", default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Defaults to 3407, matching the four-mode protocol.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Defaults to 2.",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=None,
        help="Defaults to 16384.",
    )
    parser.add_argument(
        "--draft_n_gpu_layers",
        type=int,
        default=None,
        help="Draft placement: defaults to 0 for pure_edge and -1 for pure_cloud.",
    )
    parser.add_argument(
        "--target_n_gpu_layers",
        type=int,
        default=None,
        help="Target placement: defaults to 0 for pure_edge and -1 for pure_cloud.",
    )
    parser.add_argument("--verify_num", type=int, required=True)
    parser.add_argument("--max_generated_tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=None, help="Defaults to 0.0.")
    parser.add_argument("--top_k", type=int, default=None, help="Defaults to 1.")
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Defaults to 1.0.",
    )
    parser.add_argument("--start_index_of_sample", type=int, default=0)
    parser.add_argument("--end_index_of_sample", type=int, default=163)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--target_output_tokens",
        type=int,
        default=1000,
        help="Stop after this many target-accepted draft tokens.",
    )
    parser.add_argument("--gpu_energy_device", type=int, default=0)
    parser.add_argument("--energy_sample_interval", type=float, default=0.005)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 3407
    if args.threads is None:
        args.threads = 2
    if args.ctx_size is None:
        args.ctx_size = 16384
    if args.temp is None:
        args.temp = 0.0
    if args.top_k is None:
        args.top_k = 1
    if args.top_p is None:
        args.top_p = 1.0
    return args


def main():
    args = parse_arguments()
    payload, result_path = LocalSpeculativeEvaluator(args).run()
    summary = payload["summary"]
    print(f"result: {result_path}")
    print(
        "mode={mode} dataset={dataset} tokens={tokens} TPT={tpt:.3f}ms "
        "throughput={throughput:.3f}token/s energy/100={energy}".format(
            mode=args.mode,
            dataset=args.dataset,
            tokens=summary["actual_accepted_draft_tokens"],
            tpt=summary["weighted_tpt_ms"],
            throughput=summary["throughput_tokens_per_second"],
            energy=summary["model_energy_joules_per_100_tokens"],
        )
    )


if __name__ == "__main__":
    main()

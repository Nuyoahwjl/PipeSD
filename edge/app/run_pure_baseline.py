#!/usr/bin/env python3
"""Run a standalone pure-edge or pure-cloud autoregressive baseline."""

import argparse
import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))

from src.pure_baseline import PureBaselineEvaluator


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Pure single-model baseline. Run pure_edge with the draft model on "
            "the edge CPU, or pure_cloud with the target model on the cloud GPU."
        )
    )
    parser.add_argument("--mode", required=True, choices=("pure_edge", "pure_cloud"))
    parser.add_argument("--dataset", required=True, choices=("humaneval", "gsm8k"))
    parser.add_argument("--model_path", default="")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--output_root", default="exp/exp__wjl")
    parser.add_argument("--result_tag", default="four_mode_s1_paper")
    parser.add_argument("--run_id", default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Defaults to 1234, matching both Edge and Cloud roles.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Defaults to 2 for pure_edge and 1 for pure_cloud.",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=None,
        help="Defaults to 16384 for pure_edge and 1024 for pure_cloud.",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=None,
        help="Defaults to 0 for pure_edge and -1 for pure_cloud.",
    )
    parser.add_argument("--max_generated_tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=None, help="Role-matched default: 0.0.")
    parser.add_argument("--top_k", type=int, default=None, help="Role-matched default: 1.")
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Defaults to 0.95 for pure_edge and 1.0 for pure_cloud.",
    )
    parser.add_argument("--start_index_of_sample", type=int, default=0)
    parser.add_argument("--end_index_of_sample", type=int, default=163)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--target_output_tokens", type=int, default=1000)
    parser.add_argument("--gpu_energy_device", type=int, default=0)
    parser.add_argument("--energy_sample_interval", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_arguments()
    payload, result_path = PureBaselineEvaluator(args).run()
    summary = payload["summary"]
    print(f"result: {result_path}")
    print(
        "mode={mode} dataset={dataset} tokens={tokens} TPT={tpt:.3f}ms "
        "throughput={throughput:.3f}token/s energy/100={energy}".format(
            mode=args.mode,
            dataset=args.dataset,
            tokens=summary["actual_output_tokens"],
            tpt=summary["weighted_tpt_ms"],
            throughput=summary["throughput_tokens_per_second"],
            energy=summary["model_energy_joules_per_100_tokens"],
        )
    )


if __name__ == "__main__":
    main()

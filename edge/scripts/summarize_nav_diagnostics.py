import argparse
import json
from pathlib import Path
from statistics import mean


def load_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(rows, field):
    values = []
    for row in rows:
        diagnostics = row.get("diagnostics", {})
        values.extend(diagnostics.get(field, []))
    return values


def mean_or_none(values):
    if not values:
        return None
    return float(mean(values))


def histogram(values):
    hist = {}
    for value in values:
        key = str(value)
        hist[key] = hist.get(key, 0) + 1
    return hist


def summarize_rows(rows):
    draft_lengths = flatten(rows, "draft_lengths")
    accepted_lengths = flatten(rows, "accepted_lengths")
    rejected_lengths = flatten(rows, "rejected_lengths")

    total_time = sum(row["total_time"] for row in rows)
    total_output_tokens = sum(row["output_length"] for row in rows)
    total_verifications = sum(row["verify_stats"]["num_verifications"] for row in rows)

    return {
        "samples": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "avg_total_time": float(mean(row["total_time"] for row in rows)),
        "weighted_time_per_output_token": float(total_time / total_output_tokens) if total_output_tokens else None,
        "avg_output_length": float(mean(row["output_length"] for row in rows)),
        "avg_num_verifications": float(mean(row["verify_stats"]["num_verifications"] for row in rows)),
        "verification_frequency": float(total_verifications / total_output_tokens) if total_output_tokens else None,
        "mean_verify_spec_len": mean_or_none(draft_lengths),
        "mean_accept_len": mean_or_none(accepted_lengths),
        "mean_rejected_len": mean_or_none(rejected_lengths),
        "rollback_rate": float(sum(1 for value in rejected_lengths if value > 0) / len(rejected_lengths)) if rejected_lengths else 0.0,
        "acceptance_rate": float(sum(accepted_lengths) / sum(draft_lengths)) if draft_lengths else None,
        "draft_length_hist": histogram(draft_lengths),
        "accepted_length_hist": histogram(accepted_lengths),
        "rejected_length_hist": histogram(rejected_lengths),
    }


def build_paths(args):
    base = Path("exp/exp__gsm") / args.dataset
    tag = f"_tag={args.result_tag}" if args.result_tag else ""
    bw = f"{args.bandwidth:g}"
    return {
        "vanilla": base / "vanilla" / f"gamma_{args.vanilla_gamma}{tag}_bw={bw}MB.json",
        "hsl": base / "hsl" / f"st={args.hsl_thresh}{tag}_bw={bw}MB.json",
        "pipesd": base / "pipesd" / f"st={args.pipesd_single_thresh}_mt={args.pipesd_multi_thresh}_merge=dp{tag}_bw={bw}MB.json",
        "edgeLLM": base / "edgeLLM" / f"edgeLLM_alpha={args.edgellm_init_alpha}_mult={args.edgellm_multiply_times}{tag}_bw={bw}MB.json",
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize speculative decoding diagnostics across baselines.")
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--bandwidth", type=float, default=2.5)
    parser.add_argument("--result_tag", default="nav_diag_pilot")
    parser.add_argument("--vanilla_gamma", type=int, default=6)
    parser.add_argument("--hsl_thresh", type=float, default=0.99)
    parser.add_argument("--pipesd_single_thresh", type=float, default=0.9)
    parser.add_argument("--pipesd_multi_thresh", type=float, default=0.95)
    parser.add_argument("--edgellm_init_alpha", type=float, default=0.92)
    parser.add_argument("--edgellm_multiply_times", type=float, default=0.95)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    summaries = {}
    for algorithm, path in build_paths(args).items():
        rows = load_rows(path)
        summaries[algorithm] = {
            "path": str(path),
            **summarize_rows(rows),
        }

    print(json.dumps(summaries, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

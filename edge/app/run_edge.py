# eval_humaneval.py
import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import time
import json
from pathlib import Path
from multiprocessing import Queue

from src.util import seed_everything, parse_arguments
from src.engine import Decoding
import torch.multiprocessing as mp

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

# 1. 更改类名，使其更贴切
class CloudEdgeSpeculativeEval(Decoding):
    def __init__(self, args):
        # 将队列的创建也移到这里，作为类的属性
        self.request_queue = Queue()
        self.response_queue = Queue()
        super().__init__(args)
        self.samples = self.load_data()

    def _limit_samples_for_debug(self, samples):
        max_samples = getattr(self.args, "max_samples", None)
        if max_samples is None:
            return samples
        return samples[:max(0, max_samples)]

    def load_data(self):
        """
        从文件中加载并返回数据样本。
        """
        self.color_print(f"[Main] Loading data ({self.args.dataset}) from: {self.args.data_path}", 3)
        try:
            with open(self.args.data_path, "r", encoding="utf-8") as f:
                raw_samples = [json.loads(line) for line in f]

            dataset = getattr(self.args, "dataset", "").lower()

            if dataset == "humaneval":
                samples = self._load_humaneval_samples(raw_samples)
            elif dataset == "gsm8k":
                samples = self._load_gsm8k_samples(raw_samples)
            else:
                self.color_print(f"[Main] 未知数据集 {self.args.dataset}，默认使用 humaneval 解析格式。", 1)
                samples = self._load_humaneval_samples(raw_samples)

            samples = samples[self.start_index_of_sample:self.end_index_of_sample+1]
            samples = self._limit_samples_for_debug(samples)
            self.color_print(f"[Main] Loaded {len(samples)} samples.", 3)
            return samples
        except Exception as e:
            print(f"[Main] Error loading data: {e}")
            return [] # 如果加载失败，返回空列表以避免崩溃

    def _load_humaneval_samples(self, raw_samples):
        samples = []
        for idx, item in enumerate(raw_samples):
            prompt = item.get("prompt")
            if prompt is None:
                self.color_print(f"[Main] 跳过第 {idx} 个样本：缺少 prompt 字段", 1)
                continue
            task_id = item.get("task_id", item.get("question_id", idx))
            samples.append({"prompt": prompt, "task_id": task_id})
        return samples

    def _load_gsm8k_samples(self, raw_samples):
        samples = []
        for idx, item in enumerate(raw_samples):
            question = item.get("question")
            if question is None:
                self.color_print(f"[Main] 跳过第 {idx} 个样本：缺少 question 字段", 1)
                continue
            # GSM8K 不包含显式 id，这里按顺序分配 task_id
            samples.append({"prompt": question, "task_id": idx})
        return samples

    def preprocess(self, input_text):
        dataset = getattr(self.args, "dataset", "").lower()
        if dataset == "mt_bench":
            turns = input_text.get("turns", [])
            # 使用首轮问题作为生成前缀，并附带类别便于区分任务
            prompt = turns[0].strip()
            if input_text.get("category"):
                prompt = f"[{input_text['category']}] {prompt}"
            raw_task_id = input_text.get("task_id", input_text.get("question_id", 0))
        elif dataset == "gsm8k":
            prompt = input_text["prompt"].strip()
            raw_task_id = input_text.get("task_id", input_text.get("question_id", 0))
        else:
            prompt = input_text['prompt'].strip()
            raw_task_id = input_text['task_id']

        try:
            task_id = int(str(raw_task_id).split('/')[-1])
        except Exception:
            task_id = raw_task_id
        task_id_offset = getattr(self.args, "task_id_offset", 0)
        if isinstance(task_id, int):
            task_id += task_id_offset
        return prompt, task_id

    def postprocess(self, input_text, output_text):
        bos_token = '<s>'
        eos_token = '</s>'
        if output_text.startswith(bos_token):
            generation = output_text[len(input_text)+len(bos_token)+1:]
        else:
            generation = output_text[len(input_text):]

        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                generation = generation[:generation.index(stop_word)].strip()

        return input_text + "\n    " + generation.replace("\t", "    ")

    def _append_bayes_record(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.load(path.open("r", encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
        existing.append(record)
        with path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def _run_latency_trial(self, thresh_single: float, thresh_multi: float, tokens_per_sample: int):
        # BO candidates must be evaluated independently.  Formal evaluation
        # intentionally keeps cumulative token durations, so reset the shared
        # buffer only at the BO-trial boundary.
        self._token_durations = []
        self.args.verify_thresh_single = thresh_single
        self.args.verify_thresh_multi = thresh_multi
        if tokens_per_sample <= 0:
            raise ValueError("bayes_tokens_per_sample must be positive")

        # Evaluate every selected sample with the same accepted-token budget.
        # The objective is the token-weighted mean over all collected latencies;
        # e.g. 10 samples x 20 tokens = 200 tokens per BO candidate.
        for sample in self.samples:
            sample_start = len(self._token_durations)
            no_progress = 0
            while len(self._token_durations) - sample_start < tokens_per_sample:
                before = len(self._token_durations)
                prompt, task_id = self.preprocess(sample)
                self._reset_state()
                remaining = tokens_per_sample - (before - sample_start)
                self.edge_process_draft_model(
                    prompt,
                    task_id,
                    persist_result=False,
                    max_accepted_tokens=remaining,
                )
                if len(self._token_durations) == before:
                    no_progress += 1
                    if no_progress >= 2:
                        raise RuntimeError(
                            f"BO trial could not collect accepted tokens from task {task_id}."
                        )
                else:
                    no_progress = 0
        return list(self._token_durations)

    def bayes_optimize_thresholds(self):
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Bayesian optimization requires scikit-optimize (`skopt`). "
                "Install it or run without --bayes_optimize."
            ) from exc
        if not self.samples:
            self.color_print("[Main] 无样本可用于贝叶斯优化，直接退出。", 1)
            return
        tokens_target = getattr(self.args, "bayes_tokens_per_sample", None)
        if tokens_target is None:
            tokens_target = getattr(self.args, "bayes_tokens_per_trial", None)
        if tokens_target is None:
            tokens_target = 20
        log_path = Path(self.exp_name) / "bayes_trials.json"
        search_space = [
            Real(self.args.bayes_single_min, self.args.bayes_single_max, name="verify_thresh_single"),
            Real(self.args.bayes_multi_min, self.args.bayes_multi_max, name="verify_thresh_multi"),
        ]

        def objective(params):
            thresh_single, thresh_multi = params
            latencies = self._run_latency_trial(thresh_single, thresh_multi, tokens_target)
            if not latencies:
                return float("inf")
            avg_latency = float(sum(latencies) / len(latencies))
            record = {
                "thresh_single": thresh_single,
                "thresh_multi": thresh_multi,
                "avg_token_time": avg_latency,
                "num_tokens": len(latencies),
                "num_samples": len(self.samples),
                "tokens_per_sample": tokens_target,
                "latencies": latencies,
            }
            self._append_bayes_record(log_path, record)
            self.color_print(
                f"[Bayes] st={thresh_single:.4f}, mt={thresh_multi:.4f}, tokens={len(latencies)}, avg={avg_latency:.6f}",
                3,
            )
            return avg_latency

        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=getattr(self.args, "bayes_calls", 16),
            n_initial_points=getattr(self.args, "bayes_init_points", 1),
            acq_func="EI",
            xi=getattr(self.args, "bayes_ei_xi", 0.1),
            random_state=self.seed,
        )
        best_single, best_multi = result.x
        best_record = {
            "best_thresh_single": best_single,
            "best_thresh_multi": best_multi,
            "best_avg_token_time": float(result.fun),
        }
        self._append_bayes_record(log_path, {"best": best_record})
        self.color_print(
            f"[Bayes] 最优阈值: st={best_single:.4f}, mt={best_multi:.4f}, avg={result.fun:.6f}",
            2,
        )
        self.args.verify_thresh_single = float(best_single)
        self.args.verify_thresh_multi = float(best_multi)
        self.verify_thresh_single = float(best_single)
        self.verify_thresh_multi = float(best_multi)
        return best_record

    @torch.no_grad()
    def eval(self):
        if getattr(self.args, "bayes_optimize", False):
            if self.verify_strategy != "hybrid":
                self.color_print("[Main] 当前策略非 hybrid，将自动切换为 hybrid 进行阈值搜索。", 3)
                self.verify_strategy = "hybrid"
            self.bayes_optimize_thresholds()
            if getattr(self.args, "bayes_only", False):
                self.color_print("[Main] BO 完成，已按 --bayes_only 跳过正式评测。", 2)
                return
            self.color_print("[Main] BO 完成，使用最优阈值继续正式评测。", 2)
        # start_time = time.time()
        seed_everything(self.args.seed)
        for i in range(1):
            for sample in self.samples:
                prompt, task_id = self.preprocess(sample)
                self._reset_state()
                bandwidth_label = f"{self.bandwidth_MBps:g}"
                # bandwidth_label = "{0}"
                path = self.exp2path(bandwidth_label)

                if os.path.exists(path):
                    print('path exists:', path)
                    with open(path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            data = []
                    if any(entry['task_id'] == task_id for entry in data):
                        self.color_print(f"[Main] Task {task_id} already evaluated. Skipping.", 3)
                        continue
                    
                
                self.edge_process_draft_model(prompt, task_id)
        

if __name__ == "__main__":
    MAX_RETRY = 5
    for i in range(MAX_RETRY):
        try:
            args = parse_arguments()
            evaluator = CloudEdgeSpeculativeEval(args)
            evaluator.eval()
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            if i == MAX_RETRY - 1:
                raise
            time.sleep(2)

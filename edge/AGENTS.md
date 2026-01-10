# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the PEARL runtime: `engine.py` manages draft/target interplay, `speculator.py` contains verification logic, and `util.py` stores local model/data paths—update it before running anything.
- `benchmark/` bundles evaluation entry points (HumanEval, GSM8K, MT-Bench, MGSM) whose outputs land in `logs/`.
- `scripts/` keeps reproducible pipelines (`run_para_sd.sh`, `run_ar.sh`, ablations, case studies) and should mirror the commands cited in the paper.
- `exp/` captures experiment snapshots/comparisons, `static/` carries demo assets, and `applications.py` is the optional UI harness.

## Build, Test, and Development Commands
- ```sh
  sh install.sh
  ```
  Provision the Python/torch/accelerate stack; rerun whenever dependencies change.
- ```sh
  sh scripts/run_para_sd.sh
  ```
  Launch the canonical PEARL pipeline and capture metrics in `logs/`.
- ```sh
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1 -e H_PSD_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b
  ```
  Template for custom experiments; adjust dataset flags, temperatures, and experiment IDs.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indents, snake_case modules/functions, UpperCamelCase dataclasses, and keep tensors typed/annotated near their creation.
- Favor short, deterministic helpers instead of stateful globals; document tricky scheduling math directly above the block.
- Run `ruff check` (installed via `install.sh`) or `python -m compileall src` before committing; keep multiline argument lists sorted and trailing commas for tidy diffs.

## Testing Guidelines
- Run `python scripts/test_bandwidth_limiter.py` after touching throughput code; it exercises the limiter described in `BANDWIDTH_LIMIT.md`.
- For algorithmic checks, execute the matching `benchmark/eval_*.py` script in both `para_sd` and `ar` modes, diff the JSON summaries in `logs/`, and attach the delta to your PR.
- Name fresh exports `<DATASET>_<MODE>_<MODEL>.json` and stash longer narratives in `exp/<experiment>/README.md` so results stay reproducible.

## Commit & Pull Request Guidelines
- Mirror the current short, imperative commit style (`fix draft sync`, `engine: add post verify`), aiming for ≤60 characters.
- PRs should outline purpose, reproduction command, and any notable metrics or UI screenshots; link to issues/experiments when applicable.
- Split env/bootstrap tweaks from algorithm or benchmark changes to keep bisects readable.

## Security & Configuration Tips
- Keep proprietary checkpoints outside the repo and reference them via absolute (or symlinked) paths in `src/util.py`.
- Do not commit API keys, SSH configs, or cluster hostnames; rely on ignored `.env` files.
- Validate new configs on a throttled GPU first—`BANDWIDTH_LIMIT.md` shows how to cap throughput so shared clusters stay responsive.

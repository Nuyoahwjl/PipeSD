# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements PEARL (Parallel Speculative Decoding with Adaptive Draft Length), a parallel inference framework for Large Language Models (LLMs) that achieves significant speedups while maintaining lossless generation quality. PEARL uses pre-verify and post-verify strategies to achieve adaptive draft length based on verification results.

## Key Features

- Up to 3.87× speedup on HumanEval, 3.81× on GSM8K, 3.59× on MT-bench, and 3.95× on MGSM
- Provably lossless generation
- Training-free and requires no additional memory
- Compatible with draft-then-verify frameworks like EAGLE and Medusa
- Supports GGUF models through llama-cpp-python

## Code Architecture

### Core Components

1. **src/engine.py** - Main decoding engine implementing various decoding strategies:
   - Autoregressive sampling (`autoregressive_sampling`)
   - Standard speculative decoding (`speculative_decoding`)
   - Parallel speculative decoding (`parallel_speculative_decoding`)
   - Variants without specific strategies

2. **src/util.py** - Utility functions for:
   - Argument parsing and model configuration
   - Random seed management
   - Logits normalization and sampling
   - Probability operations (top-k, top-p filtering)

3. **src/kvcache*.py** - KV Cache implementations for different scenarios:
   - Basic KV cache (`kvcache.py`)
   - Batching support (`kvcache_batching.py`)
   - GGUF model support (`kvcache_gguf.py`)

4. **benchmark/** - Evaluation scripts for different datasets:
   - HumanEval (`eval_humaneval.py`)
   - GSM8K (`eval_gsm8k.py`)
   - MT-Bench (`eval_mt_bench.py`)

### Decoding Strategies

1. **Autoregressive Decoding** - Standard token-by-token generation
2. **Speculative Decoding** - Uses draft model to generate speculative tokens, verified by target model
3. **Parallel Speculative Decoding (PEARL)** - Parallel implementation with adaptive draft length using pre-verify/post-verify strategies

## Common Development Tasks

### Running Tests

```bash
# Run a simple test for speculative decoding with GGUF models
python test_speculative_decoding_gguf.py

# Run auto-regressive sampling test
python test_autoregressive_sampling.py
```

### Running Benchmarks

```bash
# Parallel speculative decoding with different model combinations
sh scripts/run_para_sd.sh

# Standard speculative decoding
sh scripts/run_sd.sh

# Auto-regressive decoding
sh scripts/run_ar.sh
```

Example command for running evaluation:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py --eval_mode para_sd --gamma 5 -n 1 -e H_PSD_codellama_7_70b --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
```

### Key Parameters

- `--eval_mode`: Decoding strategy (small, large, sd, para_sd, etc.)
- `--gamma`: Number of speculative tokens
- `--draft_model`: Path to draft model
- `--target_model`: Path to target model
- `--max_tokens`: Maximum tokens to generate
- `--temp`: Temperature for sampling

## Build and Development Commands

### Installation

```bash
sh install.sh
```

This installs required packages including:
- PyTorch 2.1.2 with CUDA 12.1 support
- Transformers, accelerate, numpy, and other dependencies
- llama-cpp-python for GGUF model support (optional)

### Environment Setup

1. Update model paths in `src/util.py` (lines 31-38 and 49)
2. Update data paths as needed
3. Ensure models are available at specified paths

## Testing

The project includes test scripts:
- `test_speculative_decoding_gguf.py` - Test speculative decoding with GGUF models
- `test_autoregressive_sampling.py` - Test autoregressive sampling

Run tests with:
```bash
python test_speculative_decoding_gguf.py
python test_autoregressive_sampling.py
```

## Common Issues and Solutions

1. **AttributeError: 'list' object has no attribute 'get_seq_length'**
   - Solution: In latest transformers (>=4.49.0), change `past_key_values[0][0].shape[2]` to `past_key_values.get_seq_length()`

2. **Unexpected generations or meaningless text**
   - Solution: Add `.to(torch.float32)` to prevent precision overflow (e.g., line 187 of `src/engine.py`)

3. **GGUF model support issues**
   - Ensure llama-cpp-python is installed: `pip install llama-cpp-python`
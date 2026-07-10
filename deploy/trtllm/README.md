# Real NVFP4 W4A4KV4 inference on TensorRT-LLM

The main release ships ReQAT models in fake-quantized form (BF16 weights,
quantized at runtime) for accuracy evaluation via `vllm_custom`. This directory
adds the hardware-native NVFP4 path on Blackwell (B200 / SM100) with
TensorRT-LLM to measure **end-to-end inference throughput speedup** of the
trained W4A4KV4 model.

## Scope

This path targets throughput measurement. Its KV cache is stock TensorRT-LLM's
per-block-scaled **E2M1**, while ReQAT trains with an **E1M2** KV cache plus a
post-RoPE **K-shift**, so task accuracy on this path may differ slightly from the
`vllm_custom` path (`inference.py`), which is the one used for accuracy
evaluation. The E1M2/K-shift difference is softmax-invariant and does not affect
throughput. The exported `k_shift.pt` and `--kv affine` are provided for a
TRT-LLM kernel that supports affine KV.

## Requirements

A Blackwell GPU with native NVFP4 support. On non-Blackwell hardware
TensorRT-LLM cannot run the FP4 path.

The KV cache format is hardware-dependent. B200 / GB200 (SM100) have the
TRTLLM-gen FP4 attention kernel and run W4A4KV4 (`--kv nvfp4`). DGX Spark / GB10
(SM121) lack that kernel, so use FP8 KV — W4A4KV8 (`--kv fp8`).

```bash
python3.10 -m venv trtllm_env && . trtllm_env/bin/activate
apt-get install -y libopenmpi-dev libpython3.10
pip install tensorrt-llm==1.2.1 --extra-index-url https://pypi.nvidia.com
```

Use a dedicated venv; tensorrt-llm pins its own torch.

## 1. Export a released ReQAT checkpoint to real NVFP4

```bash
python deploy/trtllm/export_nvfp4.py \
    --model superdocker/R1-Llama-8B-ReQAT-nvfp4-w4a4kv4-fake \
    --output_dir ./R1-Llama-8B-ReQAT-nvfp4 \
    --kv nvfp4
```

Detects a released ReQAT (fake) checkpoint and loads its QAT'd weights into the
base architecture (or quantizes a plain BF16 model directly). Writes an HF-layout
checkpoint with `hf_quant_config.json` (`quant_algo=NVFP4`,
`kv_cache_quant_algo=NVFP4`) that loads in TRT-LLM and vLLM
(`--quantization modelopt_fp4`). `--kv` selects `nvfp4` / `affine` / `fp8` / `none`.

## 2. Smoke-run inference

```bash
python deploy/trtllm/run_trtllm.py --model ./R1-Llama-8B-ReQAT-nvfp4 \
    --prompt "What is 12*13?" --max_new_tokens 256
```

## 3. Measure end-to-end throughput speedup

`bench_throughput.py` builds a synthetic workload and runs `trtllm-bench`
(adaptive concurrency, memory-bound batching), then prints output token
throughput. NVFP4's smaller weight + KV footprint admits a larger concurrent
batch, which is where the speedup comes from at long generation lengths.

```bash
# NVFP4 (ReQAT)
python deploy/trtllm/bench_throughput.py \
    --model ./R1-Llama-8B-ReQAT-nvfp4 \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --num_requests 1024 --input_len 512 --output_len 16384 \
    --output_json nvfp4.json

# BF16 baseline (same architecture)
python deploy/trtllm/bench_throughput.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --num_requests 1024 --input_len 512 --output_len 16384 \
    --output_json bf16.json
```

Speedup = NVFP4 `OUTPUT_THROUGHPUT_TOK_S` / BF16 `OUTPUT_THROUGHPUT_TOK_S`. The
gain grows with generation length, since longer sequences make the BF16 KV cache
the binding memory constraint. Absolute numbers depend on the GPU, model,
sequence lengths, request count, and TensorRT-LLM version.

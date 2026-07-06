# Real NVFP4 W4A4KV4 inference on TensorRT-LLM

The main release ships ReQAT models in fake-quantized form (BF16 weights,
quantized at runtime) for accuracy evaluation via `vllm_custom`. This directory
adds the hardware-native NVFP4 deployment path on Blackwell (B200 / SM100) with
TensorRT-LLM, so the trained W4A4KV4 model runs — and its 4-bit inference
speedup is measured — end to end.

## Q-FIT at inference

- **Pre-RoPE paired scaling** is folded into the `q_proj` / `k_proj` weights
  during DAROC (`methods/daroc/auto_scale.py::merge_scales_to_weights`); the
  exported NVFP4 weights carry it with no runtime op.
- **Post-RoPE K-shift** subtracts a static per-`(kv_head, head_dim)` constant
  from every key, so it is softmax-invariant and has no throughput effect. It
  only reshapes the KV distribution to cut FP4 error, and reproducing it exactly
  on-device needs an affine (biased) KV quantizer plus the e1m2 KV format the
  paper uses — neither is in stock TRT-LLM (per-block-scaled e2m1). The exported
  `k_shift.pt` and `--kv affine` support that path; the speed benchmark uses
  plain NVFP4 KV, which is representative of runtime cost.

Speed is therefore equivalent to a standard ModelOpt NVFP4 W4A4KV4 deployment.

## Environment

```bash
python3.10 -m venv trtllm_env && . trtllm_env/bin/activate
apt-get install -y libopenmpi-dev libpython3.10
pip install tensorrt-llm==1.2.1 --extra-index-url https://pypi.nvidia.com
```

Use a dedicated venv; tensorrt-llm pins its own torch.

## Export

```bash
python deploy/trtllm/export_nvfp4.py \
    --model superdocker/R1-Llama-8B-ReQAT-nvfp4-w4a4kv4-fake \
    --out   ./R1-Llama-8B-ReQAT-nvfp4 \
    --kv nvfp4
```

Detects a released ReQAT (fake) checkpoint and loads its QAT'd weights into the
base architecture, or quantizes a plain BF16 model directly. Writes an HF-layout
checkpoint with `hf_quant_config.json` (`quant_algo=NVFP4`,
`kv_cache_quant_algo=NVFP4`) that loads in TRT-LLM and vLLM
(`--quantization modelopt_fp4`). `--kv` selects `nvfp4` / `affine` / `fp8` / `none`.

## Run

```bash
python deploy/trtllm/run_trtllm.py --model ./R1-Llama-8B-ReQAT-nvfp4 \
    --prompt "What is 12*13?" --max-tokens 256
```

## Benchmark

```bash
python deploy/trtllm/bench_trtllm.py --model ./R1-Llama-8B-ReQAT-nvfp4 --tag nvfp4 \
    --batch 1 8 --in-len 128 --out-len 512
python deploy/trtllm/bench_trtllm.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --tag bf16 \
    --batch 1 8 --in-len 128 --out-len 512
```

Speedup = bf16 `ms_per_tok` / nvfp4 `ms_per_tok`.

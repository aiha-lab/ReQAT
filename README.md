# ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training

This repository provides the core implementation for the paper:

> **ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training**

> [!NOTE]
> **Hardware requirements.** The experiments in the paper were conducted on 8× NVIDIA H200 GPUs. Running this codebase requires a Hopper-generation GPU (H100/H200) or above, as it relies on NVFP4 hardware support and related kernel optimizations.
>
> **Models and Dataset.** This repository was tested by fine-tuning **DeepSeek-R1-Distill-Qwen-14B** and **DeepSeek-R1-Distill-Llama-8B** on the math subset of **open-thoughts/OpenThoughts3-1.2M**. Before Stage-2 QAT, please verify that Stage-1 BF16 SFT improves reasoning accuracy on your target benchmarks.
>
> **Scope of this release.** This repository provides the core implementation of the paper's methods — including all main algorithmic components (TAQ, SEM, Q-FIT), model definitions, training objectives, and inference code. However, we have not re-run the full set of experiments due to resource constraints. The code is released to facilitate reproducibility and future research, not as a verified reproduction package.

## ReQAT Models

Pre-trained ReQAT models are available on HuggingFace:

| Model | Format | HuggingFace |
|---|---|---|
| R1-Qwen-14B | NVFP4 W4A16 | [superdocker/R1-Qwen-14B-ReQAT-nvfp4-w4a16-fake](https://huggingface.co/superdocker/R1-Qwen-14B-ReQAT-nvfp4-w4a16-fake) |
| R1-Qwen-14B | NVFP4 W4A4KV4 | [superdocker/R1-Qwen-14B-ReQAT-nvfp4-w4a4kv4-fake](https://huggingface.co/superdocker/R1-Qwen-14B-ReQAT-nvfp4-w4a4kv4-fake) |
| R1-LLaMA-8B | NVFP4 W4A4KV4 | [superdocker/R1-Llama-8B-ReQAT-nvfp4-w4a4kv4-fake](https://huggingface.co/superdocker/R1-Llama-8B-ReQAT-nvfp4-w4a4kv4-fake) |

These models are released in fake-quantized format, compatible with the `vllm_custom` inference code in this repository.

> [!NOTE]
> For real (hardware-native) NVFP4 quantization, applying the K-shift at inference time requires a modification to TensorRT-LLM. This is not included in the current release.

## Reasoning Benchmark

To evaluate a ReQAT model on reasoning benchmarks:

```bash
bash scripts/inference/inference.sh <GPU_IDs>
# e.g., bash scripts/inference/inference.sh 0,1,2,3
```

This runs `inference.py` across AIME-2025, AIME-90, and MATH-500 with temperature 0.6 / top-p 0.95. The script can also be invoked directly for more control:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ENABLE_THINKING=true python -m inference \
    --model superdocker/R1-Qwen-14B-ReQAT-nvfp4-w4a4kv4-fake \
    --dataset AIME-90 \
    --temperature 0.6 \
    --top_p 0.95 \
    --seed 42
```

Supported datasets: `AIME-2024`, `AIME-2025`, `AIME-90`, `MATH-500`, `GSM8K`, `NuminaMath-1.5`, `GPQA-Diamond`, `LiveCodeBench`.

`inference.py` loads fake-quantized models via `vllm_custom` (registered through `register_fake_quantized_models()`), which maps the `architectures` field in the model config to the custom vLLM model class.

## Training Pipeline

### Overview

ReQAT is a reasoning-centric QAT framework for W4A4KV4 deployment of large reasoning models (LRMs). It addresses the failure mode that FP4 quantization errors concentrate at low-entropy token positions (digits, operators), where sampling errors cascade through reasoning traces. ReQAT consists of three components:

- **TAQ** (Trace-Aligned QAT): two-stage QAT that revisits identical reasoning traces across BF16 FT and QAT stages
- **SEM** (Selective Entropy Minimization): auxiliary loss that reinforces model confidence at low-entropy positions
- **Q-FIT** (Quantization-Friendly Initialization via Transformation): calibrates pre-RoPE scaling and post-RoPE key shift to reduce KV cache quantization error before QAT

```
Dataset Prep  →  Stage 1: BF16 SFT  →  Q-FIT Calibration  →  Stage 2: QAT + SEM  →  Inference
```

### Step 0: Dataset Preparation (for TAQ)

TAQ implements trace-aligned training by using an identical fixed dataset for both Stage 1 and Stage 2. The dataset is pre-generated once and reused:

```bash
cd training/reqat/dataset
python gen_openthought_dataset.py
# Saves: dataset/OpenThought3-DeepSeek-89k-math-sft
```

`gen_openthought_dataset.py` filters the OpenThoughts3-1.2M dataset to the math domain with boxed answers, samples 89K examples, and tokenizes them with DeepSeek-R1 format. Both `sft.sh` and `qat.sh` point to the same saved dataset, ensuring Stage-2 QAT revisits the same reasoning traces encountered during Stage-1 BF16 FT.

### Step 1: Stage 1 BF16 Fine-Tuning

```bash
cd training/reqat
bash scripts/sft.sh
```

Standard SFT with cross-entropy loss on the fixed dataset. Uses ZeRO-3 via Accelerate (`configs/zero3.yaml`) with full linear fine-tuning (`configs/sft_full_linears.yaml`). Entry point: `sft.py`, trainer: `QATSFTTrainer` from `modelopt.torch.quantization.plugins`.

### Step 2: Q-FIT Calibration (DAROC)

```bash
bash scripts/quantization/daroc.sh <GPU_ID> <sft_model_path>
```

Q-FIT is implemented in `methods/daroc/`. It jointly calibrates pre-RoPE paired scaling and post-RoPE key shift to minimize KV cache quantization error prior to QAT.

**Implementation** (`methods/daroc/auto_scale.py`, `optimize_pre_rope_scaling()`):
- Captures pre-RoPE Q, K, V projections layer-by-layer via forward hooks
- Performs a 20×20 grid search over scale exponent α_s ∈ [0, 1] and shift fraction α_m ∈ [0, 1]
- At each grid point, computes attention score MSE between BF16 reference and KV-quantized output
- Selects (α_s, α_m) minimizing MSE

**Outputs:**
- Calibrated model with pre-RoPE scales folded into Q/K projection weights
- `k_shift_dict.pt`: per-layer post-RoPE key shift vectors (m in the paper), applied during QAT via `KShiftCallback`

### Step 3: Stage 2 QAT with SEM

```bash
cd training/reqat
bash scripts/qat.sh
```

Quantization-aware training using NVFP4 W4A4KV4 fake-quantization (`NVFP4_W4A4_E1M2_KV4_FAKE_CFG`) on the same dataset as Stage 1.

**SEM** is implemented in `training/reqat/custom_trainer.py` (`SEMTrainer`). It augments the standard cross-entropy loss with a weighted entropy minimization term:

```
L_SEM = L_SFT + λ · (1/T) Σ_t w_t H_t
```

where H_t is the predictive entropy at step t and the soft weight w_t emphasizes low-entropy positions:

```
w_t = max(0, 1 − (H_t − H_min) / (τ − H_min + ε))
```

**K-Shift** during QAT is applied via `KShiftCallback` (`training/reqat/k_shift_attention.py`). The callback fires at `on_train_begin` (after ModelOpt installs quantizers) and patches `_QuantAttention.forward` to subtract the calibrated shift m from post-RoPE key states.

Key training arguments:
```bash
--sem                          # enable SEM loss
--entropy_threshold 0.75       # τ = 75th percentile
--lambda_sem 0.1               # λ weighting
--shift_k true                 # enable K-shift
--k_shift_path <model>/k_shift_dict.pt
--quant_cfg NVFP4_W4A4_E1M2_KV4_FAKE_CFG
```

After QAT, `convert_fake_mxfp4_weight_only.py` converts the fake-quantized model to an evaluation-ready format compatible with the repository's inference code.

## Repository Structure

```
├── training/                   # Stage 1 SFT and Stage 2 QAT
│   ├── reqat/
│   │   ├── sft.py              # main training script
│   │   ├── custom_trainer.py   # SEMTrainer (SEM loss)
│   │   ├── k_shift_attention.py # KShiftCallback (Q-FIT K-shift at QAT time)
│   │   ├── convert_fake_mxfp4_weight_only.py  # post-QAT conversion
│   │   ├── dataset/
│   │   │   └── gen_openthought_dataset.py  # TAQ dataset preparation
│   │   ├── configs/            # Accelerate and training configs
│   │   └── scripts/
│   │       ├── sft.sh          # Stage 1 script
│   │       └── qat.sh          # Stage 2 script
│   └── modelopt/               # Custom NVIDIA ModelOpt fork
│       └── torch/quantization/ # NVFP4_W4A4_E1M2_KV4_FAKE_CFG and QATSFTTrainer
├── methods/
│   ├── daroc/                  # Q-FIT calibration
│   │   ├── run_daroc.py        # entry point
│   │   ├── pre_quant.py        # layer-wise calibration loop
│   │   ├── auto_scale.py       # joint scale/shift grid search
│   │   └── ...
│   └── utils/
│       └── data_utils.py       # calibration data loading
├── vllm_custom/                # Custom vLLM model classes for inference
├── scripts/
│   ├── quantization/daroc.sh   # Q-FIT calibration script
│   └── inference/inference.sh
├── inference.py
└── get_dataset.py
```

## Custom ModelOpt Package

`training/modelopt/` is a custom fork of [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) that adds:

- `NVFP4_W4A4_E1M2_KV4_FAKE_CFG`: quantization config for NVFP4 W4A4 with E1M2 KV cache
- `QATSFTTrainer`: HuggingFace Trainer integration for fake-quantized QAT

## Citation
This repository implements the core code based on “Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models” (COLM 2025), using NVIDIA ModelOpt as the quantization framework.

```bibtex
@article{liu2025quantization,
  title={Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models},
  author={Liu, Ruikang and Sun, Yuxuan and Zhang, Manyi and Bai, Haoli and Yu, Xianzhi and Yu, Tiezheng and Yuan, Chun and Hou, Lu},
  journal={arXiv preprint arXiv:2504.04823},
  year={2025}
}

@misc{nvidia_modelopt,
  title        = {{NVIDIA Model Optimizer}},
  author       = {{NVIDIA}},
  year         = {2026},
  howpublished = {\url{https://github.com/NVIDIA/Model-Optimizer}},
  note         = {Accessed: 2026-06-14}
}

@inproceedings{MLSYS2024_42a452cb,
 author = {Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
 booktitle = {Proceedings of Machine Learning and Systems},
 editor = {P. Gibbons and G. Pekhimenko and C. De Sa},
 pages = {87--100},
 title = {AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration},
 url = {https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf},
 volume = {6},
 year = {2024}
}
```

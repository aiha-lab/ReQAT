# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied and Adapted from https://github.com/huggingface/gpt-oss-recipes/blob/main/sft.py
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stage 1 SFT:
  accelerate launch --config_file configs/zero3.yaml sft.py \
      --config configs/sft_full_linears.yaml \
      --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
      --dataset_name <dataset>

Stage 2 QAT with SEM:
  accelerate launch --config_file configs/zero3.yaml sft.py \
      --config configs/sft_full_linears.yaml \
      --sem \
      --entropy_threshold 0.75 \
      --lambda_sem 0.1 \
      --model_name_or_path <sft_checkpoint> \
      --shift_k true \
      --k_shift_path <sft_checkpoint>/k_shift_dict.pt \
      --quant_cfg NVFP4_W4A4_E1M2_KV4_FAKE_CFG \
      --dataset_name <dataset>
"""

import torch
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    TrlParser,
)
from utils import (
    get_original_huggingface_quant_method,
    get_peft_config_for_moe,
    is_distributed_job,
    load_dataset_from_hub_or_local,
)

import modelopt.torch.opt as mto
from modelopt.torch.quantization.plugins import QATSFTTrainer, QuantizationArguments
from modelopt.torch.utils import print_rank_0
from k_shift_attention import KShiftCallback

# Enable automatic save/load of modelopt state with HuggingFace checkpointing
mto.enable_huggingface_checkpointing()


@dataclass
class CustomArguments:
    cache_dir: str | None = field(default=None)
    sem: bool = field(
        default=False,
        metadata={"help": "QAT with Selective Entropy Minimization (SEM)."},
    )
    lambda_sem: float = field(
        default=0.01,
        metadata={"help": "Weight for the SEM loss term."},
    )
    sem_only: bool = field(
        default=False,
        metadata={"help": "Use SEM loss only (no cross-entropy)."},
    )
    entropy_threshold: float = field(
        default=0.25,
        metadata={"help": "Fraction of lowest-entropy tokens to apply EM loss on."},
    )
    shift_k: bool = field(
        default=False,
        metadata={"help": "Apply K-Shift to Qwen2 attention after RoPE."},
    )
    k_shift_path: str = field(
        default=None,
        metadata={"help": "Path to k_shift tensors (.pt). Dict[int, Tensor] mapping layer_idx to k_shift."},
    )


def main(script_args, training_args, model_args, quant_args):
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": model_args.torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if get_original_huggingface_quant_method(model_args.model_name_or_path) == "mxfp4":
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    if not is_distributed_job():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if custom_args.shift_k and custom_args.k_shift_path is None:
        raise ValueError("--k_shift_path is required when --shift_k is True")

    dataset = load_dataset_from_hub_or_local(script_args, training_args)

    # Select trainer class
    custom_kwargs = dict()
    if custom_args.sem:
        from custom_trainer import SEMTrainer
        trainer_cls = SEMTrainer
        custom_kwargs["entropy_threshold"] = custom_args.entropy_threshold
        custom_kwargs["lambda_sem"] = custom_args.lambda_sem
        custom_kwargs["sem_only"] = custom_args.sem_only
    else:
        trainer_cls = QATSFTTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config_for_moe(model, model_args),
        quant_args=quant_args,
        **custom_kwargs,
    )

    if custom_args.shift_k:
        trainer.add_callback(KShiftCallback(k_shift_path=custom_args.k_shift_path))
        print_rank_0(f"[K-Shift] Callback added — will apply k_shift at training start (post-quantization)")

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, QuantizationArguments, CustomArguments))
    script_args, training_args, model_args, quant_args, custom_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, quant_args)

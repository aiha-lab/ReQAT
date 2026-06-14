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

import argparse
import gc
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, AutoConfig
from utils import get_original_huggingface_quant_method

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

def _load_checkpoint_state_dict(model_path):
    """Load state dict from checkpoint (supports safetensors and pytorch formats)."""
    import os
    from glob import glob
    
    # Try safetensors first
    safetensor_files = glob(os.path.join(model_path, "*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(load_file(f))
        return state_dict
    
    # Try pytorch format
    bin_files = glob(os.path.join(model_path, "pytorch_model*.bin"))
    if bin_files:
        state_dict = {}
        for f in bin_files:
            state_dict.update(torch.load(f, map_location="cpu"))
        return state_dict
    
    # Try single file
    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        from safetensors.torch import load_file
        return load_file(single_file)
    
    single_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(single_file):
        return torch.load(single_file, map_location="cpu")
    
    return None


def _to_oai_mxfp4_weight_only(model, quantizer, args=None, cfg=None):
    new_state_dict = {}

    for name, param in model.state_dict().items():
        if "proj.weight" in name:
            if 'int' in args.cfg: # Reset quantizer everytime
                quantizer = TensorQuantizer()
                quantizer.set_from_attribute_config(cfg)
            if args.exclude_string is not None:
                if args.exclude_string in name:
                    print(f'Skipping {name}')
                    new_state_dict[name] = param
                else:
                    print(f'Quantizing {name}')
                    qparam = quantizer(param)
                    new_state_dict[name] = qparam
            else:
                print(f'Quantizing {name}')
                qparam = quantizer(param)
                new_state_dict[name] = qparam
        else:
            new_state_dict[name] = param

    return new_state_dict


def convert_and_save(model, tokenizer, output_path, quantizer, args=None, cfg=None):
    # Convert weights to mxfp4
    quantized_state_dict = _to_oai_mxfp4_weight_only(model, quantizer, args, cfg)

    # Save converted weights
    model.save_pretrained(output_path, state_dict=quantized_state_dict)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model_path", type=str, help="path to the fake-quantized model from QAT.")

    parser.add_argument(
        "--lora_path",
        type=str,
        help="path to the LoRA-QAT adapter weights. You can only specify lora_path or model_path, not both.",
    )

    parser.add_argument(
        "--base_path",
        type=str,
        help="path to the base model used for LoRA-QAT. Only used if lora_path is specified.",
    )

    parser.add_argument(
        "--output_path", type=str, required=True, help="location to save converted model."
    )

    parser.add_argument(
        "--cfg", type=str, required=True, help=""
    )

    parser.add_argument(
        "--activation_cfg", type=str, default='none', required=False, help=""
    )

    parser.add_argument(
        "--kv_cfg", type=str, default='none', required=False, help=""
    )
    parser.add_argument(
        "--kv_quant", type=str, default='false', required=False, help=""
    )
    parser.add_argument(
        "--exclude_string", type=str, default=None, required=False, help=""
    )
    parser.add_argument(
        "--shift_k", type=str, default='false', required=False, help=""
    )
    parser.add_argument(
        "--k_shift_path", type=str, default=None, required=False, help=""
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    kwargs = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
    if args.lora_path:
        assert args.model_path is None, "You can only specify lora_path or model_path, not both."
        model_path = args.base_path
        if get_original_huggingface_quant_method(args.base_path) == "mxfp4":
            kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    else:
        model_path = args.model_path

    if args.cfg=='mxfp4':
        cfg = {'num_bits': (2, 1), 'block_sizes': {-1: 32, 'type': 'dynamic', 'scale_bits': (8, 0)}, 'enable': True, 'pass_through_bwd': True}
    elif args.cfg=='nvfp4':
        cfg = {'num_bits': (2, 1), 'block_sizes': {-1: 16, 'type': 'dynamic', 'scale_bits': (4, 3)}, 'axis': None, 'enable': True, 'pass_through_bwd': True}
    elif args.cfg=='none':
        pass
    else:
        raise NotImplementedError

    if args.cfg=='none':
        cfg = None
        quantizer = lambda x: x
    else:
        quantizer = TensorQuantizer()
        quantizer.set_from_attribute_config(cfg)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if args.k_shift_path is not None:
        assert args.shift_k=='true'
        from k_shift_attention import (                                                    
            apply_k_shift_to_model,                                                     
            load_k_shifts,                                                      
            check_k_shift_stats,                                                             
        ) 
        print(f"[K-Shift] Loading k_shifts from {args.k_shift_path}")
        k_shifts = load_k_shifts(args.k_shift_path, device="cpu")
        num_modified = apply_k_shift_to_model(model, k_shifts)
        print(f"[K-Shift] Applied k_shift to {num_modified} attention layers")

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()  # Merge LoRA-QAT adapter weights to base model
        torch.cuda.empty_cache()
        gc.collect()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize and save model
    convert_and_save(model, tokenizer, args.output_path, quantizer, args, cfg)

    # Activation quantize and config update
    if args.activation_cfg!='none' or args.kv_quant!='false':
        config = AutoConfig.from_pretrained(model_path)
        is_kv_quant = args.kv_quant=='true'
        
        # Determine architecture name
        if 'llama' in args.model_path:
            arch_name = "LlamaFakeQuantizedModelOPTForCausalLM"
        else:
            arch_name = "Qwen2FakeQuantizedModelOPTForCausalLM"
        
        # Build fake_quant_config
        fake_quant_config = {
            "cfg": args.activation_cfg, 
            "kv_quant": is_kv_quant,
            "shift_k": args.shift_k=='true'
        }

        if args.kv_cfg!='none':
            fake_quant_config['kv_cfg'] = args.kv_cfg
        
        config.update({
            "architectures": [arch_name],
            "fake_quant_config": fake_quant_config
        })
        config.save_pretrained(args.output_path)
        
        print(f"\n[Config] Saved with architecture: {arch_name}")

    print('Done')

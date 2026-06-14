import os
import argparse

import transformers
from .pre_quant import run_pre_rope_scaling, apply_pre_rope_scaling
from .quantizer import pseudo_quantize_model_weight
import torch.nn as nn
import torch


def save_pre_rope_model(args):
    # load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cpu"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model.eval()

    # run pre-rope scaling calibration
    pre_rope_results = run_pre_rope_scaling(
        model,
        tokenizer,
        x_bit=args.x_bits,
        w_bit=args.w_bits,
        q_config=args.q_config,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        calib_data=args.calib_data,
        model_path=args.model,
    )

    # save model
    if args.save_qmodel_path:
        os.makedirs(args.save_qmodel_path, exist_ok=True)
        tokenizer.save_pretrained(args.save_qmodel_path)
        # load model weights
        model_transformers = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="cpu"
        )
        apply_pre_rope_scaling(model_transformers, pre_rope_results)
        k_shift_dict = dict()
        for k, v in model.state_dict().items():
            if 'k_shift' in k:
                k_shift_dict[k] = v.clone()
        torch.save(k_shift_dict, f'{args.save_qmodel_path}/k_shift_dict.pt')
        model_transformers.save_pretrained(args.save_qmodel_path)
        if 'llama' in args.model:
            model.config.architectures = ["LlamaFakeQuantizedModelOPTForCausalLM"]
        else:
            model.config.architectures = ["Qwen2FakeQuantizedModelOPTForCausalLM"]
        model.config.fake_quant_config = {
            "cfg": 'nvfp4',
            "kv_quant": True,
            "shift_k": True,
            "kv_cfg": 'nvfp4_e1m2',
        }
        model.config.save_pretrained(args.save_qmodel_path)
        print(f"Model saved at {args.save_qmodel_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path of the hf model")
    # quantization config
    parser.add_argument("--x_bits", type=int, default=None)
    parser.add_argument("--w_bits", type=int, default=None)
    parser.add_argument("--w_groupsize", type=int, default=-1)
    parser.add_argument("--w_asym", action="store_true", help="disable zero_point")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--calib_data", type=str, default="pileval")
    # apply/save/load pre-rope scaling
    parser.add_argument(
        "--save_qmodel_path", type=str, default=None, help="save the pre-rope scaling results"
    )
    args = parser.parse_args()

    # get quantization config (apart from w_bit)
    q_config = {
        "zero_point": args.w_asym,  # by default False
        "q_group_size": args.w_groupsize,  # whether to use group quantization
    }
    args.q_config = q_config
    print(args)

    save_pre_rope_model(args)


if __name__ == "__main__":
    main()


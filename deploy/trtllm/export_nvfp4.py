import argparse
import copy
import glob
import os

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, Qwen2ForCausalLM)

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

BASE_ARCH = {"llama": LlamaForCausalLM, "qwen2": Qwen2ForCausalLM}
KV_CFG = {"nvfp4": mtq.NVFP4_KV_CFG, "affine": mtq.NVFP4_AFFINE_KV_CFG, "fp8": mtq.FP8_KV_CFG}


def quant_cfg(kv):
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    if kv != "none":
        for k, v in KV_CFG[kv]["quant_cfg"].items():
            if "bmm_quantizer" in k:
                cfg["quant_cfg"][k] = v
    return cfg


def load_model(model, device):
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    is_fake = hasattr(config, "fake_quant_config")
    if is_fake:
        config.architectures = None
        del config.fake_quant_config
        config.use_cache = True
        m = BASE_ARCH[config.model_type].from_pretrained(
            model, config=config, dtype=torch.bfloat16,
            device_map=device if device == "cuda" else None)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16,
            device_map=device if device == "cuda" else None, trust_remote_code=True)
    if device != "cuda":
        m = m.to(device)
    return m.eval(), is_fake


def save_k_shift(model, out):
    local = snapshot_download(model, ignore_patterns=["*.pt", "*.bin"])
    shifts = {k: v for f in glob.glob(os.path.join(local, "*.safetensors"))
              for k, v in load_file(f).items() if "k_shift" in k}
    if shifts:
        os.makedirs(out, exist_ok=True)
        torch.save(shifts, os.path.join(out, "k_shift.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--kv", type=str, choices=["nvfp4", "affine", "fp8", "none"], default="nvfp4")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, is_fake = load_model(args.model, args.device)
    if is_fake:
        save_k_shift(args.model, args.output_dir)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t) > 200][: args.n_samples]

    def forward_loop(m):
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=args.seqlen).input_ids.to(args.device)
            m(ids)

    model = mtq.quantize(model, quant_cfg(args.kv), forward_loop)
    mtq.print_quant_summary(model)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

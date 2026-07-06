import argparse
import json
import time

import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import LLM, CudaGraphConfig, KvCacheConfig


def build_llm(model, max_batch_size, max_seq_len):
    cuda_graph = CudaGraphConfig(
        batch_sizes=[2 ** i for i in range(max_batch_size.bit_length())],
        max_batch_size=max_batch_size, enable_padding=True)
    return LLM(model=model, backend="pytorch", tensor_parallel_size=1,
               enable_chunked_prefill=True,
               kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7),
               cuda_graph_config=cuda_graph,
               max_batch_size=max_batch_size, max_seq_len=max_seq_len)


def run(llm, batch, in_len, out_len, iters=3):
    sp = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
    prompts = [list(range(1, in_len + 1)) for _ in range(batch)]

    llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    torch.cuda.synchronize()

    best = None
    for _ in range(iters):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)

    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    return {"batch": batch, "in_len": in_len, "out_len": out_len,
            "wall_s": round(best, 4), "tok_s": round(gen / best, 1),
            "ms_per_tok": round(1000.0 * best / (gen / batch), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=512)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    llm = build_llm(args.model, max(args.batch), args.in_len + args.out_len + 8)
    rows = []
    for bs in args.batch:
        row = run(llm, bs, args.in_len, args.out_len)
        row["tag"] = args.tag
        print(json.dumps(row))
        rows.append(row)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()

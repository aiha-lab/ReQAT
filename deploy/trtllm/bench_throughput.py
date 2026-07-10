import argparse
import json
import os
import random
import subprocess
import sys


def write_dataset(path, num_requests, input_len, output_len):
    random.seed(0)
    with open(path, "w") as f:
        for i in range(num_requests):
            ids = [random.randint(1000, 100000) for _ in range(input_len)]
            f.write(json.dumps({"task_id": i, "input_ids": ids, "output_tokens": output_len}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--num_requests", type=int, default=1024)
    parser.add_argument("--input_len", type=int, default=512)
    parser.add_argument("--output_len", type=int, default=8192)
    parser.add_argument("--kv_cache_fraction", type=float, default=0.9)
    parser.add_argument("--output_json", type=str, default="trtllm_bench.json")
    args = parser.parse_args()

    stem = os.path.splitext(args.output_json)[0]
    dataset = stem + "_dataset.jsonl"
    report = stem + "_report.json"
    write_dataset(dataset, args.num_requests, args.input_len, args.output_len)

    trtllm_bench = os.path.join(os.path.dirname(sys.executable), "trtllm-bench")
    if os.path.isdir(args.model):
        model_args = ["--model", args.base_model or args.model, "--model_path", args.model]
    else:
        model_args = ["--model", args.model]
    subprocess.run([
        trtllm_bench, *model_args,
        "throughput", "--dataset", dataset, "--backend", "pytorch",
        "--kv_cache_free_gpu_mem_fraction", str(args.kv_cache_fraction),
        "--num_requests", str(args.num_requests), "--report_json", report,
    ], check=True)

    perf = json.load(open(report))["performance"]
    print(f"OUTPUT_THROUGHPUT_TOK_S {perf['system_output_throughput_tok_s']:.1f}  ({args.model})")


if __name__ == "__main__":
    main()

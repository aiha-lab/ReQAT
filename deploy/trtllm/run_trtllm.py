import argparse

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import LLM, KvCacheConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str,
                        default="What is 12*13? Solve step by step, then give the final answer.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    llm = LLM(model=args.model, backend="pytorch", tensor_parallel_size=1,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7))
    sp = SamplingParams(max_tokens=args.max_new_tokens,
                        temperature=max(args.temperature, 1e-3), top_p=args.top_p)
    out = llm.generate([args.prompt], sampling_params=sp, use_tqdm=False)[0]
    print(out.outputs[0].text)


if __name__ == "__main__":
    main()

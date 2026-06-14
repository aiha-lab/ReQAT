import os
import shutil
from huggingface_hub import snapshot_download

DATASETS = {
    "wikitext": "wikitext",
    "pile-val-backup": "mit-han-lab/pile-val-backup",
    "NuminaMath-1.5": "AI-MO/NuminaMath-1.5",
    "AIME90": "xiaoyuanliu/AIME90",
    "aime_2025": "yentinglin/aime_2025",
    "MATH-500": "HuggingFaceH4/MATH-500",
    "gsm8k": "openai/gsm8k",
    "gpqa": "Idavidrein/gpqa",
    "code_generation_lite": "livecodebench/code_generation_lite",
}

os.makedirs("./datasets", exist_ok=True)

for dataset, repo in DATASETS.items():
    save_path = f"./datasets/{dataset}"

    if os.path.exists(save_path):
        files = os.listdir(save_path)
        ignored = {"readme.md", ".cache"}
        real_files = [f for f in files if f.lower() not in ignored]

        if not real_files:
            print(f"Only README.md or .cache found for {dataset}, redownloading...")
            shutil.rmtree(save_path)
        else:
            print(f"Skipping {dataset} (already exists with proper files)")
            continue

    print(f"Downloading {dataset} from {repo}...")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=save_path,
        token=os.environ.get("HF_TOKEN", None)
    )

#!/bin/bash
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions_base

datasets=("AIME-2025" "AIME-90" "MATH-500")
devices=$1      # 0,1,2,3
for dataset in "${datasets[@]}"; do
for seed in 42; do
model_path=superdocker/R1-Qwen-14B-ReQAT-nvfp4-w4a16-fake
NVIDIA_VISIBLE_DEVICES=${devices} CUDA_VISIBLE_DEVICES=${devices} ENABLE_THINKING=true \
python -m inference \
    --model $model_path \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --dataset $dataset \
    --seed $seed
echo $model
done
done

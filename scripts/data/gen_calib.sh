#!/bin/bash

model=$1    # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
devices=$2  # 0,1,2,3

model_name=$(basename "$model")

CUDA_VISIBLE_DEVICES=${devices} \
python inference.py \
    --model $model \
    --dataset NuminaMath-1.5 \
    --max_samples 256 \
    --output_dir ./datasets/gen_data/${model_name}

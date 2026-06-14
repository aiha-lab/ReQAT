export WANDB_PROJECT='NVFP4-QAT'
export HF_HUB_OFFLINE=0
base_dir=${PWD}

for lr in 1e-5; do
for epoch in 0.25; do
output_dir=r1-qwen-14b-sft-$lr-ot3-math-89k-epoch$epoch
export WANDB_NAME=$output_dir
cd $base_dir && accelerate launch --config_file configs/zero3.yaml sft.py \
    --config configs/sft_full_linears.yaml \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --min_lr_rate 0.01 \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --dataset_name dataset/OpenThought3-DeepSeek-89k-math-sft \
    --output_dir $output_dir
done
done

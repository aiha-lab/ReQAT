export WANDB_PROJECT='NVFP4-QAT'
export HF_HUB_OFFLINE=0
base_dir=${PWD}

for lr in 1e-5; do
for epoch in 0.0625; do
model_path=r1-qwen-14b-sft-1e-5-ot3-math-89k-epoch0.25-daroc
output_dir=r1-qwen-14b-nvfp4-w4a4-e1m2-kv4-sem-from-sft$from-daroc-ot3-math-89k-$lr-epoch$epoch-lambda0.1-threshold0.75
export WANDB_NAME=$output_dir
cd $base_dir && accelerate launch --config_file configs/zero3.yaml sft.py \
    --config configs/sft_full_linears.yaml \
    --sem \
    --entropy_threshold 0.75 \
    --lambda_sem 0.1 \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --min_lr_rate 0.01 \
    --model_name_or_path $model_path \
    --shift_k true \
    --k_shift_path $model_path/k_shift_dict.pt \
    --dataset_name dataset/OpenThought3-DeepSeek-89k-math-sft \
    --quant_cfg NVFP4_W4A4_E1M2_KV4_FAKE_CFG \
    --output_dir $output_dir
cd $base_dir && python convert_fake_mxfp4_weight_only.py --model_path $output_dir --output_path $output_dir-fakequant --cfg nvfp4 --activation_cfg nvfp4 --kv_quant true --kv_cfg nvfp4_e1m2 --shift_k true --k_shift_path $model_path/k_shift_dict.pt
done
done

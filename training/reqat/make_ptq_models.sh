for model in r1-qwen-14b-sft-1e-5-ot3-math-89k-epoch0.25; do
    python convert_fake_mxfp4_weight_only.py --model_path $model --output_path $model-nvfp4-fakequant --cfg nvfp4 --activation_cfg nvfp4 --kv_quant true
done

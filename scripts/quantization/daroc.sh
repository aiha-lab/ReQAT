model=$2
CUDA_VISIBLE_DEVICES=$1 python -m methods.daroc.run_daroc \
    --model $model \
    --n_samples 256 \
    --seqlen 512 \
    --save_qmodel_path $model-daroc

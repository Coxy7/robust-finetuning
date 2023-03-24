trap "exit" INT

LOG_DIR="$HOME/log/robust_ft"
DATA_DIR="$HOME/data"
MODEL_DIR="$HOME/model"
EXP_NAME="release"
NUM_WORKERS=6
basic_args="--code_dir ./ --data_dir $DATA_DIR --log_dir $LOG_DIR --exp_name $EXP_NAME"

mode=$1; shift

model=$1; shift
model_args="--arch $model --load_pretrained $MODEL_DIR/clip"
model=${model//[-\/@]/_}  # avoid '/' in filename (e.g. ViT-B/32 -> ViT_B_32)

methods=$1; shift
masking=$1; shift

seed=$1; shift
seed_args="--seed $seed --data_split_seed $seed --run_number $seed"

do_train () {
    script_name=$1; shift
    num_devices=$(python -c 'import torch; print(torch.cuda.device_count())')
    port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    OMP_NUM_THREADS=2 \
    torchrun --nnodes=1 --nproc_per_node=$num_devices --rdzv_endpoint="localhost:$port" \
        "./$script_name.py" $basic_args $model_args $seed_args \
        --num_workers $NUM_WORKERS --num_iters_trainset_test 0 "$@"
}
do_test () {
    num_devices=$(python -c 'import torch; print(torch.cuda.device_count())')
    port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    OMP_NUM_THREADS=2 \
    torchrun --nnodes=1 --nproc_per_node=$num_devices --rdzv_endpoint="localhost:$port" \
        ./eval.py  $basic_args $model_args $seed_args --load_run_number $seed \
        --num_workers $NUM_WORKERS "$@"
}
dummy(){ unused(){ :;} }

if [[ $mode = 'train' ]]; then
    train=do_train
    test=dummy
elif [[ $mode = 'eval' ]]; then
    train=dummy
    test=do_test
elif [[ $mode = 'train-eval' ]]; then
    train=do_train
    test=do_test
else
    echo "Incorrect script mode '$mode', should be: train / eval / train-eval"
    exit 1
fi


for method in $methods; do

    zeroshot_run_name="$model-zeroshot"
    wise_run_name="$model-WiSE-FT"
    teacher_run_name=$zeroshot_run_name

    if [[ $masking = '' ]]; then
        RUN_NAME="$model-$method"
    else
        RUN_NAME="$model-$method-$masking"
    fi

    if [[ $method = 'zeroshot' ]]; then
        RUN_NAME=$zeroshot_run_name
        $train 'main_standard' --run_name $RUN_NAME \
            --epoch 1 --num_iters_train 0 \
            --num_iters_trainset_test 0 --lr_schedule none \
            "$@"
    elif [[ $method = 'LP' ]]; then     # linear probe
        $train 'main_standard' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            --freeze_backbone --weight_decay 0 \
            "$@"
    elif [[ $method = 'FT' ]]; then     # end-to-end fine-tune
        $train 'main_standard' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            "$@"
    elif [[ $method = 'WiSE-FT' ]]; then
        RUN_NAME=$wise_run_name
        $train 'main_standard' --run_name $RUN_NAME \
            --epoch 1 --num_iters_train 0 \
            --num_iters_trainset_test 0 --lr_schedule none \
            --load_run_name "$model-FT" --load_run_number $seed \
            --wise_base_run_name $zeroshot_run_name \
            "$@"
    elif [[ $method = 'FT_KD' ]]; then
        $train 'main_distill' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            --teacher_run_name $teacher_run_name \
            --task std --distill_mode kd \
            "$@"
    elif [[ $method = 'FT_KD_image_mask' ]]; then
        $train 'main_distill' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            --teacher_run_name $teacher_run_name \
            --task std --distill_mode kd_image_mask \
            --distill_masking $masking \
            "$@"
    elif [[ $method = 'FT_FD' ]]; then
        $train 'main_distill' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            --teacher_run_name $teacher_run_name \
            --task std --distill_mode fd \
            "$@"
    elif [[ $method = 'FT_FD_image_mask' ]]; then
        $train 'main_distill' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            --teacher_run_name $teacher_run_name \
            --task std --distill_mode fd_image_mask \
            --distill_masking $masking \
            "$@"
    elif [[ $method = 'FT_FD_mae_mask' ]]; then
        $train 'main_distill' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            --teacher_run_name $teacher_run_name \
            --task std --distill_mode fd_mae_mask \
            --distill_masking $masking \
            "$@"
    elif [[ $method = 'FT_FD_attn_mask' ]]; then
        $train 'main_distill' --run_name $RUN_NAME \
            --load_run_name $zeroshot_run_name \
            --teacher_run_name $teacher_run_name \
            --task std --distill_mode fd_attn_mask \
            --distill_masking $masking \
            "$@"
    fi

    $test --run_name "eval-$RUN_NAME" --load_run_name $RUN_NAME "$@"

done

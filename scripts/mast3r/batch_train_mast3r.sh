#!/bin/bash

# 给定路径
BASE_PATH="data/Tanks_for_mast3r"
GPU=0
MAX_GPU=8

# for SCENE in $(ls $BASE_PATH); do
#     IMG_PATH="$BASE_PATH/$SCENE/images"
#     LOG_PATH="$BASE_PATH/$SCENE.log"
#     if [ -d "$IMG_PATH" ]; then
#         CUDA_VISIBLE_DEVICES=$GPU python coarse_init_eval_mast3r.py --img_base_path $IMG_PATH 2>&1 | tee $LOG_PATH &
#         GPU=$(( (GPU + 1) % MAX_GPU ))
#     fi
# done

wait
echo "Init done"

GPU=0
PORT=7039
OUTPUT_PATH="output/Tanks_for_mast3r"
for SCENE in $(ls $BASE_PATH); do
    SCENE_PATH="$BASE_PATH/$SCENE"
    MODEL_PATH="$OUTPUT_PATH/$SCENE"
    LOG_PATH="$OUTPUT_PATH/$SCENE.log"

    if [ -d "$SCENE_PATH" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python train_joint.py -s $SCENE_PATH -m $MODEL_PATH --eval --optim_pose --port=$PORT 2>&1 | tee $LOG_PATH &
    fi
    sleep 5s
    
    PORT=$((PORT + 10))
    GPU=$((GPU + 1))
    # 如果GPU编号达到最大值，则重新从0开始
    if [ $GPU -ge $MAX_GPU ]; then
        GPU=0
    fi

done

wait
echo "Train done"

GPU=0
OUTPUT_PATH="output/Tanks_for_mast3r"
for SCENE in $(ls $BASE_PATH); do
    SCENE_PATH="$BASE_PATH/$SCENE"
    MODEL_PATH="$OUTPUT_PATH/$SCENE"

    if [ -d "$SCENE_PATH" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python render.py -s $SCENE_PATH -m $MODEL_PATH --skip_train &
    fi
    sleep 5s

    GPU=$((GPU + 1))
    # 如果GPU编号达到最大值，则重新从0开始
    if [ $GPU -ge $MAX_GPU ]; then
        GPU=0
    fi
done

wait
echo "Render done"

GPU=0
for SCENE in $(ls $BASE_PATH); do
    MODEL_PATH="$OUTPUT_PATH/$SCENE"

    CUDA_VISIBLE_DEVICES=$GPU python metrics.py -m $MODEL_PATH &
    sleep 5s

    GPU=$((GPU + 1))
    # 如果GPU编号达到最大值，则重新从0开始
    if [ $GPU -ge $MAX_GPU ]; then
        GPU=0
    fi
done

wait
echo "Calculate metrics done"
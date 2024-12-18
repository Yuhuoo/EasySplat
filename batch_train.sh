#!/bin/bash

# 给定路径
BASE_PATH="data/co3d_sim"
GPU=0
MAX_GPU=8

# 创建BASE_PATH目录
mkdir -p $BASE_PATH

for SCENE in $(ls $BASE_PATH); do
    IMG_PATH="$BASE_PATH/$SCENE/images"
    LOG_PATH="$BASE_PATH/$SCENE.log"

    if [ ! -d "$IMG_PATH" ]; then
        # echo "Skipping $SCENE as $IMG_PATH is not a directory"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU python coarse_init_eval.py --img_base_path $IMG_PATH 2>&1 | tee $LOG_PATH &
    sleep 5s

    GPU=$((GPU + 1))
    # 如果GPU编号达到最大值，则重新从0开始
    if [ $GPU -ge $MAX_GPU ]; then
        GPU=0
    fi

done

wait
echo "Init done"

GPU=0
PORT=7039
OUTPUT_PATH="output/co3d_sim"

# 创建OUTPUT_PATH目录
mkdir -p $OUTPUT_PATH

for SCENE in $(ls $BASE_PATH); do
    SCENE_PATH="$BASE_PATH/$SCENE"
    MODEL_PATH="$OUTPUT_PATH/$SCENE"
    LOG_PATH="$OUTPUT_PATH/$SCENE.log"

    if [ ! -d "$SCENE_PATH" ]; then
        # echo "Skipping $SCENE as $SCENE_PATH is not a directory"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU python train_joint.py -s $SCENE_PATH -m $MODEL_PATH --eval --optim_pose --port=$PORT 2>&1 | tee $LOG_PATH &
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
for SCENE in $(ls $BASE_PATH); do
    SCENE_PATH="$BASE_PATH/$SCENE"
    MODEL_PATH="$OUTPUT_PATH/$SCENE"

    if [ ! -d "$SCENE_PATH" ]; then
        # echo "Skipping $SCENE as $SCENE_PATH is not a directory"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU python render.py -s $SCENE_PATH -m $MODEL_PATH --eval --optim_pose --skip_train &
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
GT_COLMAP_PATH="data/Tanks"
for SCENE in $(ls $BASE_PATH); do
    MODEL_PATH="$OUTPUT_PATH/$SCENE"
    GT_POSE_PATH="$GT_COLMAP_PATH/$SCENE"

    CUDA_VISIBLE_DEVICES=$GPU python metrics.py -m $MODEL_PATH --gt_pose_path $GT_POSE_PATH  &
    sleep 5s

    GPU=$((GPU + 1))
    # 如果GPU编号达到最大值，则重新从0开始
    if [ $GPU -ge $MAX_GPU ]; then
        GPU=0
    fi
done

wait
echo "Calculate metrics done"
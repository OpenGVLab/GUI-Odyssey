#!/bin/bash
MODEL="/path/to/model" # "../OdysseyAgent"
DATA_ROOT=/path/to/chat_format_annotation #  ../data/train_anno/low_random_split.json   ../data/train_anno/high_random_split.json

exp_name=OdysseyAgent
mkdir -p output/"$exp_name"
OUTPUT_DIR=output_$exp_name
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 10001 + 40000))
GPUS=$((GPUS_PER_NODE * NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --dataset $DATA_ROOT \
    --fp16 True \
    --fix_vit True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 300 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 800 \
    --gradient_checkpointing True \
    --deepspeed finetune/ds_config_zero2.json


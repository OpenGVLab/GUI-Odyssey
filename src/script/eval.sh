#!/bin/bash
checkpoint=/path/to/checkpoint
ds=high_random_split # one of "low_app_split", "low_device_split", "low_random_split", "low_task_split" "high_app_split", "high_device_split", "high_random_split", "high_task_split"
DIR=`pwd`

exp_name=OdysseyAgent_$ds
mkdir -p output/"$exp_name"

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 30001 + 20000))
GPUS=$((GPUS_PER_NODE * NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $ds
echo $checkpoint
torchrun $DISTRIBUTED_ARGS eval_mm/evaluate_GUIOdyssey.py \
    --checkpoint $checkpoint --dataset $ds --batch-size 32 --his_len 4
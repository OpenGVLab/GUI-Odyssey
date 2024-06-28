#!/bin/bash
checkpoint=/path/to/checkpoint
ds=app_split # one of "app_split", "device_split", "random_split", "task_split"
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
    --checkpoint $checkpoint --dataset $ds --batch-size 16 --his_len 4
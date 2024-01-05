#!/bin/bash

SHARED_FS_DIR=/fsx

EXP_DIR=$1

# 512 * 1k * 400k = 200b tokens.
# 512 * 1k * 200k = 100b tokens.
# 512 * 1k * 100k = 50b tokens (default).
# 512 * 1k * 20k = 10b tokens.
TRAINING_STEPS=20000
if [ -n "${2}" ]; then
    TRAINING_STEPS=$2;
fi

NUM_EXPERTS=64
if [ -n "${3}" ]; then
    NUM_EXPERTS=$3;
fi

TOP_K=1
if [ -n "${4}" ]; then
    TOP_K=$4;
fi

LOSS_WEIGHT=0.1
if [ -n "${5}" ]; then
    LOSS_WEIGHT=$5;
fi

BATCH_SIZE=16
if [ -n "${6}" ]; then
    BATCH_SIZE=$6;
fi

##
### Pre-training for dMoE 125M parameter.
##

# MoE hyperparameters.
MOE_ARGUMENTS="\
--moe-num-experts=${NUM_EXPERTS} \
--moe-loss-weight=${LOSS_WEIGHT} \
--moe-top-k=${TOP_K}"

# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.0006 \
--min-lr 0.00006 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01 \
--optimizer adam"

WIKI_DATASET="\
1.0 \
${SHARED_FS_DIR}/data/enwiki_extract/hfbpe_gpt_training_data_text_document"

# NOTE: We don't train for enough tokens for the
# split to matter.
DATA_ARGUMENTS="\
--data-path ${WIKI_DATASET} \
--vocab-file ${SHARED_FS_DIR}/tokenizer/gpt2-vocab.json \
--merge-file ${SHARED_FS_DIR}/tokenizer/gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--bf16 \
--DDP-impl local \
--moe-expert-model-parallelism \
--no-async-tensor-model-parallel-allreduce \
--use-flash-attn"

CHECKPOINT_ARGUMENTS="\
--save-interval 100 \
--save ./${EXP_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

torchrun ${DISTRIBUTED_ARGUMENTS} \
       third_party/Megatron-LM/pretrain_gpt.py \
       ${MOE_ARGUMENTS} \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ./${EXP_DIR}/train.log

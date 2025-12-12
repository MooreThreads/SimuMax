#!/bin/bash

set -u
WORK_HOME=${WORK_HOME:-"./"}
PATCH_HOME=${PATCH_HOME:-"./"}
MEGATRON_HOME=${MEGATRON_HOME:-"./"}
EXAMPLE=${EXAMPLE:-"test"}
HOSTFILE=${HOSTFILE:-"./hostfile"}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
RDZV_ID=${RDZV_ID:-"0"}
MODEL_TYPE=${MODEL_TYPE:-"llama3_8b"}
TEST_TYPE=${TEST_TYPE:-"test"}
NUM_LAYERS=${NUM_LAYERS:-1}
DTYPE=${DTYPE:-"bf16"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output_${MODEL_TYPE}"}
set +u

if [ "${TEST_TYPE}" == "profile" ]; then
    PROFILER_SAVE_PATH=${WORK_HOME}/profiler_result_${MODEL_TYPE}_${EXAMPLE}
    rm -rf ${PROFILER_SAVE_PATH}
    echo "Profiler save path: ${PROFILER_SAVE_PATH}"

    TRAIN_ITERS=6
    WARMUP_ITERS=1
    PROFILE_ARGS=(
        --profile
        --profile-step-start 4
        --profile-step-end 6
        --use-pytorch-profiler
        --profile-ranks 0 1 2 3 4 5 6 7
    )
else
    PROFILE_ARGS=()
    TRAIN_ITERS=10
    WARMUP_ITERS=5
    echo "Train iters: ${TRAIN_ITERS}, Warmup iters: ${WARMUP_ITERS}"
fi
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export ACCELERATOR_BACKEND="cuda"
export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTHONPATH=${MEGATRON_HOME}:${PATCH_HOME}:$PYTHONPATH
# export NO_LOSS_REDUCE=1


# CHECKPOINT_PATH=$WORK_HOME/checkpoints/$EXAMPLE
# mkdir -p $CHECKPOINT_PATH

# LOG_PATH=$OUTPUT_DIR/logs/$EXAMPLE
# mkdir -p $LOG_PATH
# cp $0 $LOG_PATH/
TB_PATH=$OUTPUT_DIR/tboard/$EXAMPLE
mkdir -p $TB_PATH
# WB_PATH=$WORK_HOME/wandb/$EXAMPLE
# mkdir -p $WB_PATH



export NODE_ADDR=$(ip a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n1 | cut -d '/' -f1)
export GPUS_PER_NODE=8
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export MASTER_PORT=14388


LOG_DIR=$OUTPUT_DIR/${DTYPE}_$EXAMPLE
echo "Distributed log_dir: $LOG_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT 
    --log_dir $LOG_DIR
    --redirects ${LOG_REDIRECTS_LEVEL:-3}
)

if [ "${MODEL_TYPE}" == "llama2_7b" ]; then
    # llama2 7b
    num_layer=$NUM_LAYERS
    hidden_size=4096
    ffn_hidden_size=11008
    head_num=32
    num_query_groups=32
    vocab_size=32000
elif [ "${MODEL_TYPE}" == "llama3_8b" ]; then
    # llama3 8b
    num_layer=$NUM_LAYERS
    hidden_size=4096
    ffn_hidden_size=14336
    head_num=32
    num_query_groups=8
    vocab_size=128257
elif [ "${MODEL_TYPE}" == "llama3_70b" ]; then
    # llama3 70b
    num_layer=$NUM_LAYERS # layer_num=60, 切层6
    hidden_size=8192
    ffn_hidden_size=28672
    head_num=64
    num_query_groups=8
    vocab_size=128256
elif [ "${MODEL_TYPE}" == "llama3_405b" ]; then
    # llama3 405b
    num_layer=$NUM_LAYERS # 405切1层
    hidden_size=16384
    ffn_hidden_size=53248
    head_num=128
    num_query_groups=16
    vocab_size=128256
else
    echo "Model type ${MODEL_TYPE} not supported"
    exit 1
fi

MODEL_ARGS=(
    --num-layers ${num_layer}
    --hidden-size ${hidden_size} 
    --ffn-hidden-size ${ffn_hidden_size}
    --num-attention-heads ${head_num} 
    --group-query-attention 
    --num-query-groups ${num_query_groups}
    --seq-length 4096 
    --max-position-embeddings 4096 
    --norm-epsilon 1e-5 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --position-embedding-type rope 
    --no-position-embedding 
    --swiglu 
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --vocab-size ${vocab_size}
    # --tp-comm-overlap-ag
    # --tp_comm_overlap_rs 
    --disable-tp-comm-bulk-wgrad
    --disable-tp-comm-bulk-dgrad
    --disable-tp-comm-overlap-rs
    --disable-tp-comm-overlap-ag
    --disable-tp-comm-split-ag
    --disable-tp-comm-split-rs
)

# 244140625 1T
TRAINING_ARGS=(
    --seed 42 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE  
    # --train-samples 24414062 
    --init-method-std 0.008
    --use-mcore-models 
    --no-bias-dropout-fusion
    # --no-bias-swiglu-fusion
    --use-distributed-optimizer 
    --use-flash-attn 
    --sequence-parallel 
    --recompute-granularity full 
    --recompute-method block 
    --recompute-num-layers 0 
    --distributed-backend nccl
    --mock-data
)

# --no-bias-swiglu-fusion
# --no-rope-fusion
# --no-gradient-accumulation-fusion 
# --transformer-impl local transformer_engine
REGULARIZATION_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
)

WARMUP_STEPS=2000
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))

LEARNING_RATE_ARGS=(
    # --lr 1.5e-5 
    # --lr-decay-style cosine 
    # --lr-warmup-samples ${WARMUP_SAMPLES} 
    # --min-lr 1.5e-6 
    # --initial-loss-scale 65536 
    # --min-loss-scale 1.0 
    --initial-loss-scale 65536
    --min-loss-scale 1.0

    # 下面是ds 32b参数
    --train-iters ${TRAIN_ITERS}
    --lr-warmup-iters ${WARMUP_ITERS}
    --lr 2.4e-4
    --lr-decay-style cosine 
    --min-lr 1e-6
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE  
	--pipeline-model-parallel-size $PP_SIZE
    # --decoder-last-pipeline-num-layers 14
)

MIXED_PRECISION_ARGS=(
    --bf16 
    --attention-softmax-in-fp32 
    --no-masked-softmax-fusion 
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer 
    --split 1
)


EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 200000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TB_PATH 
)

if [ "$DTYPE" == "bf16" ]; then
    TRANSFORMER_ENGINE_ARGS=(
        --transformer-impl transformer_engine
    )
else
    TRANSFORMER_ENGINE_ARGS=(
        --transformer-impl transformer_engine
        --fp8-format hybrid
        --fp8-param-gather
    )
fi



cmd="torchrun ${DISTRIBUTED_ARGS[@]} ./pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]}
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]}
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]} \
        ${PROFILE_ARGS[@]}
    "
echo $cmd
eval $cmd

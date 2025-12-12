#!/bin/bash

set -u
WORK_HOME=${WORK_HOME:-"./"}
PATCH_HOME=${PATCH_HOME:-"./"}
MEGATRON_HOME=${MEGATRON_HOME:-"./"}
EXAMPLE=${EXAMPLE:-"test"}
HOSTFILE=${HOSTFILE:-"./hostfile"}
DATA_DIR=${DATA_DIR:-"./"}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
TOKENIZED_MODEL=${TOKENIZED_MODEL:-""}
RDZV_ID=${RDZV_ID:-"0"}
MASTER_PORT=${MASTER_PORT:-"12345"}
TEST_TYPE=${TEST_TYPE:-"test"}
NUM_LAYER=${NUM_LAYER:-"1"}
MODEL_TYPE=${MODEL_TYPE:-""}
DTYPE=${DTYPE:-"bf16"}
SCRIPT_FILE=${SCRIPT_FILE:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"./output_${MODEL_TYPE}"}
set +u


if [ "${TEST_TYPE}" == "profile" ]; then
    PROFILER_SAVE_PATH=${WORK_HOME}/profiler_result_ds_236b_${EXAMPLE}
    rm -rf ${PROFILER_SAVE_PATH}
    echo "Profiler save path: ${PROFILER_SAVE_PATH}"
    # export ENABLE_PROFILER=1
    # export PROFILER_FREQ=4
    # export PROFILER_SAVE_DIR=${PROFILER_SAVE_PATH}
    PROFILE_ARGS=(
        --profile
        --profile-step-start 4
        --profile-step-end 6
        --use-pytorch-profiler
    )
    TRAIN_ITERS=6
    WARMUP_ITERS=1
else
    PROFILE_ARGS=()
    TRAIN_ITERS=10
    WARMUP_ITERS=5
fi

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export ACCELERATOR_BACKEND="cuda"


export MOE_ROUTER_GROUP_TOPK=${MOE_ROUTER_GROUP_TOPK:-3}

export ENABLE_ZERO_BUBBLE=0 # if set 1, Enable zero_bubble

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_OPTIMIZER_PRECISION_AWARE_RECONFIG=${CPU_OPTIMIZER_PRECISION_AWARE_RECONFIG:-0}

export ENABLE_D2H_IN_PERMUTATION=0
# export NO_LOSS_REDUCE=1

export PYTHONPATH=${MEGATRON_HOME}:${PATCH_HOME}:$PYTHONPATH
echo $PYTHONPATH


# CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints/$EXAMPLE
# mkdir -p $CHECKPOINT_PATH
DATA_PATH=$DATA_DIR


# LOG_PATH=$OUTPUT_DIR/logs/$EXAMPLE
# mkdir -p $LOG_PATH
# cp $0 $LOG_PATH/
TB_PATH=$OUTPUT_DIR/tboard/$EXAMPLE
mkdir -p $TB_PATH
# WB_PATH=$OUTPUT_DIR/wandb/$EXAMPLE
# mkdir -p $WB_PATH


export NODE_ADDR=$(ip a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n1 | cut -d '/' -f1) # tail for cuda
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk -v node_addr="$NODE_ADDR" '{ranks[$1]=(FNR-1);} END {print ranks[node_addr];}' $HOSTFILE)
export MASTER_PORT=${MASTER_PORT:-12356}

LOG_DIR=${OUTPUT_DIR}/${DTYPE}_$EXAMPLE
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
echo $GPUS_PER_NODE $NUM_NODES $NODE_RANK $MASTER_ADDR $MASTER_PORT

NUM_LAYERS_MINUS_ONE=$((NUM_LAYER - 1))
NUM_LAYERS_MINUS_THREE=$((NUM_LAYER - 3))
if [ "$MODEL_TYPE" == "deepseek_v2" ]; then
    HEAD_NUM=128
    # KV_HEAD_NUM=128 # not used
    HIDDEN_SIZE=5120
    export MOE_NUM_EXPERTS=160
    MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1"
    MOE_FFN_HIDDEN_SIZE=1536
    FFN_HIDDEN_SIZE=12288
    VOCAB_SIZE=102400
    TOPK=6
    MOE_SHARED_FFN_HIDDEN_SIZE=3072
    echo "MOE_LAYER_FREQ: $MOE_LAYER_FREQ"

elif [ "$MODEL_TYPE" == "deepseek_v3" ]; then
    HEAD_NUM=128
    # KV_HEAD_NUM=128 # not used
    HIDDEN_SIZE=7168
    export MOE_NUM_EXPERTS=256
    MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1" 
    MOE_FFN_HIDDEN_SIZE=2048
    FFN_HIDDEN_SIZE=18432
    VOCAB_SIZE=129280
    TOPK=8
    MOE_SHARED_FFN_HIDDEN_SIZE=2048

elif [ "$MODEL_TYPE" == "kimi_1t" ]; then
    HEAD_NUM=64
    # KV_HEAD_NUM=64 # not used
    HIDDEN_SIZE=7168
    export MOE_NUM_EXPERTS=384
    MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1"
    MOE_FFN_HIDDEN_SIZE=2048
    FFN_HIDDEN_SIZE=18432
    VOCAB_SIZE=163840
    TOPK=8
    MOE_SHARED_FFN_HIDDEN_SIZE=2048
else    echo "Unknown model type: $MODEL_TYPE"
fi
MODEL_ARGS=(
    --num-layers ${NUM_LAYER}  # 60 ds 236b:ep8 pp16
    --hidden-size $HIDDEN_SIZE # dsv2 = 5120, v3=7168
    --num-attention-heads $HEAD_NUM  # dsv2/3=128, kimi-1T=64
    --seq-length 4096
    --max-position-embeddings 4096
    --norm-epsilon 1e-6
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --vocab-size $VOCAB_SIZE # dsv2 = 102400, v3=129280
    --ffn-hidden-size $FFN_HIDDEN_SIZE  # 12288 for dense, but for sequentialMLP, moe-ffn-hidden-size=1536 # dsv2 = 12288, v3=18432
    --position-embedding-type rope
    --no-position-embedding
    --swiglu
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    # --cross-entropy-loss-fusion
    
)


MOE_ARGS=(
    --num-experts ${MOE_NUM_EXPERTS}
    --expert-model-parallel-size $EP_SIZE
    --moe-token-dispatcher-type alltoall
    --moe-router-score-function softmax
    --moe-router-num-groups $EP_SIZE
    --moe-router-group-topk ${MOE_ROUTER_GROUP_TOPK}
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk $TOPK
    --moe-router-pre-softmax #deepseek use pre-softmax
    --moe-router-topk-scaling-factor 16 # pre-softmax need scaling
    --moe-aux-loss-coeff 3e-3
    --moe-expert-capacity-factor 1

    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE # ds_v2 = 1536, v3 = 2048
    --moe-shared-expert-intermediate-size $MOE_SHARED_FFN_HIDDEN_SIZE # ds_v2 = 3072, v3= 2048
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-pad-expert-input-to-capacity
)

# 24414062 1T
TRAINING_ARGS=(
    --seed 42
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    # --train-samples 24414062
    --init-method-std  0.006 # 0.02 in HF config, but 0.006 in the paper
    --use-mcore-models
    # --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    # --no-bias-swiglu-fusion
    --use-distributed-optimizer
    --use-flash-attn
    --sequence-parallel
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 0
    --distributed-backend nccl
    --multi-latent-attention
    --qk-layernorm
    --mock-data
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 1
)

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
)

MIXED_PRECISION_ARGS=(
    --bf16
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    # --data-path $DATA_PATH
    --tokenizer-type NullTokenizer
    # --tokenizer-model ${TOKENIZED_MODEL}
    --split 1
    #--dataloader-type mtepx  #default single
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 100000
    --eval-interval 1
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


cmd="torchrun ${DISTRIBUTED_ARGS[@]} ./pretrain_deepseekv2.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]}
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]}
        ${DATA_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${MLA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]} \
        ${PROFILE_ARGS[@]}
    "

USE_EPX=${USE_EPX:-0}

# run cmd directly
if [ $USE_EPX -eq 0 ]; then
  echo $cmd
  $cmd
  exit $?
fi

# run cmd with fault tolerance
source "${PATCH_HOME}/EXAMPLEs/deepseek-v2/deepseek-v2-lite/fault_tolerance_function.sh"
ft_training "$cmd"

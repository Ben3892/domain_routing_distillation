#!/bin/bash
set -e
ENV=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
# export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/Megatron-LM-v0.13.0:$PYTHONPATH
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/Megatron-LM-core_v0.14.0rc7:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset LD_LIBRARY_PATH

export BINDPCIE_PATH=${BINDPCIE_PATH:-""}

if [ $ENV = dsw ]; then
    MASTER_ADDR=$POD_0_IP
    MASTER_PORT=6001
    NNODES=${TRAINERS_NUM}
    NODE_RANK=${TRAINER_ID}
    GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### BASE CONFIG ###
MODEL_SIZE=$2
BATCH_SIZE=$3
GLOBAL_BATCH_SIZE=$4
LR=$5
MIN_LR=$6
SEQ_LEN=$7
PAD_LEN=$8
PR=$9
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=${10}
PP=${11}
CP=${12}
EP=${13}
SP=${14}
DO=${15}
FL=${16}
SFT=${17}
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${18}
OPTIMIZER_OFFLOAD=${19}
SAVE_INTERVAL=${20}
DATASET_PATH=${21}
VALID_DATASET_PATH=${22}
PRETRAIN_CHECKPOINT_PATH=${23}

# the following two values will not be used when SFT is true
TRAIN_TOKENS=${24}
WARMUP_TOKENS=${25}
###############################

OUTPUT_BASEPATH=${26}
NO_LOAD_OPTIM=${27}
NO_LOAD_RNG=${28}
NO_SAVE_OPTIM=${29}

# training options
MTP=${30}
KD=${31}
KD_IP=${32}
KD_PORT=${33}
LM_LOSS_WEIGHT=${34}
KD_LOSS_WEIGHT=${35}
MTP_LOSS_WEIGHT=${36}
MTP_KD_LOSS_WEIGHT=${37}
JSD_LOSS_WEIGHT=${38}
JSD_BETA=${39}
MTP_JSD_LOSS_WEIGHT=${40}
MTP_JSD_BETA=${41}

MP_DATASET_TYPE=${42}
TEMPERATURE=${43}

EVAL_INTERVAL=${44}
EVAL_ITERS=${45}

NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=1
if [ $MODEL_SIZE = A2.4B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    NUM_LAYERS=27
    # MP_PP0_LAYERS=$((NUM_LAYERS / 2))
    MP_PP0_LAYERS=6
    INTERMEDIATE_SIZE=10944
    MOE_INTERMEDIATE_SIZE=1408
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
    EXTRA_VOCAB_SIZE=2400
    PADDED_VOCAB_SIZE=102400
    KV_LORA_RANK=512
    QK_NOPE_HEAD_DIM=128
    QK_ROPE_HEAD_DIM=64
    V_HEAD_DIM=128
    ROPE_THETA=10000
    SCALE_FACTOR=40
    NUM_EXPERTS=64
    ROUTER_TOPK=6
    NUM_SHARED_EXPERTS=2
    MOE_LAYER_FREQ=1
    RMS_NORM_EPS=1e-6
        # --moe-token-dispatcher-type flex \
        # --moe-enable-deepep \
        # --moe-router-dtype=fp32 \
        # --moe-deepep-num-sms=128 \
    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-shared-expert-overlap \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --moe-layer-freq '([0]*1+[1]*26)' \
        --moe-aux-loss-coeff 1e-2 \
        --moe-per-layer-logging \
        --expert-tensor-parallel-size 1 \
        --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
        --expert-model-parallel-size ${EP} \
        --kv-lora-rank ${KV_LORA_RANK} \
        --qk-head-dim ${QK_NOPE_HEAD_DIM} \
        --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
        --v-head-dim ${V_HEAD_DIM} \
        --moe-router-load-balancing-type aux_loss"

elif [ $MODEL_SIZE = V100B ]; then
# MP_PP0_LAYERS=11
MP_PP0_LAYERS=9
MP_LAST_PP_LAYERS=8
HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=128
NUM_LAYERS=61
INTERMEDIATE_SIZE=9216
MOE_INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=467
PADDED_VOCAB_SIZE=128000
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=128
ROUTER_TOPK=8
NUM_SHARED_EXPERTS=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --moe-router-dtype=fp32 \
    --moe-deepep-num-sms=128 \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-per-layer-logging \
    --moe-aux-loss-coeff 0.0001 \
    --moe-layer-freq '([0]*3+[1]*58)' \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-expert-capacity-factor 1.25"



elif [ $MODEL_SIZE = V24B ]; then
MP_PP0_LAYERS=12
MP_LAST_PP_LAYERS=6
HIDDEN_SIZE=2880
NUM_ATTENTION_HEADS=64
NUM_LAYERS=36
INTERMEDIATE_SIZE=5184
MOE_INTERMEDIATE_SIZE=576
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=467
PADDED_VOCAB_SIZE=128000
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=128
ROUTER_TOPK=8
NUM_SHARED_EXPERTS=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --moe-router-dtype=fp32 \
    --moe-deepep-num-sms=128 \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-per-layer-logging \
    --moe-aux-loss-coeff 0.0001 \
    --moe-layer-freq '([0]*2+[1]*34)' \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-expert-capacity-factor 1.25"




elif [ $MODEL_SIZE = V59B ]; then
MP_PP0_LAYERS=14
MP_LAST_PP_LAYERS=8
HIDDEN_SIZE=3840
NUM_ATTENTION_HEADS=128
NUM_LAYERS=48
INTERMEDIATE_SIZE=6912
MOE_INTERMEDIATE_SIZE=768
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=467
PADDED_VOCAB_SIZE=128000
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=128
ROUTER_TOPK=8
NUM_SHARED_EXPERTS=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --moe-router-dtype=fp32 \
    --moe-deepep-num-sms=128 \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-per-layer-logging \
    --moe-aux-loss-coeff 0.0001 \
    --moe-layer-freq '([0]*2+[1]*46)' \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-expert-capacity-factor 1.25"



elif [ $MODEL_SIZE = V100B_PRUNED ]; then
# MP_PP0_LAYERS=11
MP_PP0_LAYERS=12
MP_LAST_PP_LAYERS=5
HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=64
NUM_LAYERS=61
INTERMEDIATE_SIZE=9216
MOE_INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=467
PADDED_VOCAB_SIZE=128000
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=80
NUM_EXPERTS=128
ROUTER_TOPK=4
NUM_SHARED_EXPERTS=1
RMS_NORM_EPS=1e-6


moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --moe-router-dtype=fp32 \
    --moe-deepep-num-sms=128 \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-per-layer-logging \
    --moe-aux-loss-coeff 0.0001 \
    --moe-layer-freq '([0]*3+[1]*58)' \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-expert-capacity-factor 1.25"
fi

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="weighted"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"
 

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations --recompute-modules core_attn mla_up_proj"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-recipe blockwise --fp8-format e4m3 \
        --fp8-param-gather \
        --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi


if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $FL = true ]; then
    attn_backend_option="--attention-backend flash"
else
    attn_backend_option="--attention-backend fused"
fi


if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    if [ -z ${MP_LAST_PP_LAYERS} ];then
        _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
        if [ $_check != 0 ]; then
            echo "With uneven pipelineing the left over layers must be divisible by left over stages."
            exit -1
        fi
        uneven_split_option=" \
            --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
        "
    elif [ ${PP} -gt 2 ]; then
        _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} - ${MP_LAST_PP_LAYERS}) % ( ${PP} - 2 ) ))
        if [ $_check != 0 ]; then
            echo "With uneven pipelineing the left over layers must be divisible by left over stages."
            exit -1
        fi
        uneven_split_option=" \
            --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS} \
            --decoder-last-pipeline-num-layers ${MP_LAST_PP_LAYERS}
        "
    else
        echo "MP_LAST_PP_LAYERS must be used when PP > 2"
        exit -1
    fi
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

load_options_extra=""
if [ "$NO_LOAD_OPTIM" = true ]; then
    load_options_extra="$load_options_extra \
            --no-load-optim"
fi
if [ "$NO_LOAD_RNG" = true ]; then
    load_options_extra="$load_options_extra \
            --no-load-rng"
fi
if [ "$NO_SAVE_OPTIM" = true ]; then
    load_options_extra="$load_options_extra \
            --no-save-optim"
fi

if [ $OPTIMIZER_OFFLOAD != false ]; then
    offload_option=" \
        --optimizer-cpu-offload \
        --use-precision-aware-optimizer \
        --optimizer-offload-fraction ${OPTIMIZER_OFFLOAD}"
fi

training_options=""
if [ $MTP = true ]; then
    training_options="$training_options \
        --mtp-num-layers 1"
fi
if [ $KD != none ]; then
    training_options="$training_options \
        --teacher-ip ${KD_IP} \
        --teacher-port ${KD_PORT}"
fi
if [ $KD = real ]; then
    training_options="$training_options \
        --use-distillation \
        --teacher-type real \
        --send-dataset-name True \
        --temperature ${TEMPERATURE}"
elif [ $KD = fake ]; then
    training_options="$training_options \
        --use-distillation \
        --teacher-type fake \
        --send-dataset-name False"
elif [ $KD = local ]; then
    training_options="$training_options \
        --use-distillation \
        --teacher-type local \
        --temperature ${TEMPERATURE}"
fi

training_weight_options=" \
        --lm-loss-weight ${LM_LOSS_WEIGHT} \
        --distill-loss-weight ${KD_LOSS_WEIGHT} \
        --jsd-loss-weight ${JSD_LOSS_WEIGHT} \
        --jsd-beta ${JSD_BETA} \
        --mtp-loss-weight ${MTP_LOSS_WEIGHT} \
        --mtp-distill-loss-weight ${MTP_KD_LOSS_WEIGHT} \
        --mtp-jsd-loss-weight ${MTP_JSD_LOSS_WEIGHT} \
        --mtp-jsd-beta ${MTP_JSD_BETA}"

if [ "$EVAL_ITERS" = -1 ]; then
    eval_options=" \
        --eval-interval ${EVAL_INTERVAL} \
        --full-validation"
else
    eval_options=" \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS}"
fi

if [ $SFT = true ]; then
    TRAIN_ITERS=${24}
    LR_WARMUP_ITERS=${25}
    LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
    PREFIX="finetune-mcore-deepseek-v2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" \
         --eod-mask-loss \
         --calculate-per-token-loss \
         --train-mode finetune"
else
    TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    PREFIX="pretrain-mcore-deepseek-v2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" \
        --eod-mask-loss \
        --train-mode pretrain"
fi

if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_option=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT"
elif [ ${MP_DATASET_TYPE} = "raw-packing" ]; then
    dataset_option=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT-PACKING"
elif [ ${MP_DATASET_TYPE} = "weighted" ]; then
    dataset_option=" \
        --weighted-config-file ${DATASET_PATH} \
        --dataloader-type weighted \
        --no-mmap-bin-files \
        --dataset PRETRAIN-WITH-WEIGHT"
elif [ ${MP_DATASET_TYPE} = "teacher" ]; then
    dataset_option=" \
        --weighted-config-file ${DATASET_PATH}  \
        --dataloader-type weighted \
        --no-mmap-bin-files \
        --dataset PT-TEACHER"
else 
    dataset_option=" \
        --data-path ${DATASET_PATH} \
        --split 99,1,0 \
        --dataset MMAP"
fi

if [ ${MP_SFT_PACKING} = true ]; then
    echo "Currently MLA do not support THD format attention, thus sequence packing can not be used..."
    packing_options=""
else
    packing_options=""
fi

if [ "${DATA_CACHE_PATH}" != none ]; then
    cache_options="--data-cache-path ${DATA_CACHE_PATH}"
else
    cache_options=""
fi

##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
#find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 4 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --padded-vocab-size ${PADDED_VOCAB_SIZE} \
        --patch-tokenizer-type DeepSeekV2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --multi-latent-attention \
        --moe-router-pre-softmax \
        --ckpt-format torch \
        --transformer-impl transformer_engine \
        --use-rope-scaling \
        --mscale-all-dim 1.0 \
        --rope-type yarn \
        --no-create-attention-mask-in-dataloader \
        --enable-experimental \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --moe-router-fusion \
        --moe-permute-fusion \
        --manual-gc \
        --manual-gc-interval 100 \
        "
        # --profile \
        # --profile-ranks 0 40 \
        # --use-pytorch-profiler \
        # --profile-step-start 4 \
        # --profile-step-end 5 \

        # SWA
        # --use-every-k-full-attention \
        # --sliding-window 1024 \
        # --num-k-full-attn 3 \
        # --softmax-type learnable \

        # 暂时不用
        # --no-check-for-nan-in-loss-and-grad \

run_cmd="$BINDPCIE_PATH python -m torch.distributed.run $DISTRIBUTED_ARGS  projects/deepseek/training_deepseek.py
 ${megatron_options} ${eval_options} ${training_options} ${training_weight_options} ${dataset_option} ${pr_options} ${load_options} ${load_options_extra} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} ${packing_options} ${uneven_split_option} \
 ${attn_backend_option} "

ps aux | grep "training_deepseek.py" | awk -F ' ' '{print $2}' | xargs kill 2>/dev/null || echo "Killing running processes of training_deepseek.py"
ps aux | grep "upload_ckpts.py" | awk -F ' ' '{print $2}' | xargs kill 2>/dev/null || echo "Killing running processes of upload_ckpts.py"
ps aux | grep "fake_utilization.py" | awk -F ' ' '{print $2}' | xargs kill 2>/dev/null || echo "Killing running processes of fake_utilization.py"

CKPT_AFS_PATH="output_path_cygnus/model_save/${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-$(date +%Y%m%d%H%M)"
setsid nohup python tools/upload_ckpts.py $SAVED_PRETRAIN_CHECKPOINT_PATH $CKPT_AFS_PATH $NODE_RANK 2 &

echo ${run_cmd}
eval ${run_cmd}


set +x
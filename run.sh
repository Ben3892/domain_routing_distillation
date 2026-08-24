#!/bin/bash
# 用于启动分布式深度学习训练任务，支持单机和多机模式
set -x  # 开启命令执行跟踪，便于调试
queue_h_flag=$1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 判断运行模式：独立模式 vs 分布式模式
if [[ ${IS_STANDALONE:-1} -eq 1 ]]; then
    # 独立模式：单机运行，不使用 mpirun
    mpirun=""
elif [ ${queue_h_flag} -eq 1 ]; then
    # H卡配置分布式模式：多机运行，配置 MPI 环境
    # 从 hostfile 获取主节点地址
    master=`head -1 ${TRAIN_WORKSPACE}/hostfile |awk '{print $1}'`
    
    # 配置 OpenMPI 运行参数
    mpirun="orterun
    --allow-run-as-root
    -tag-output
    -timestamp-output
    --hostfile ${TRAIN_WORKSPACE}/hostfile
    --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME
    -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_bond_0
    -x PATH
    -x LD_LIBRARY_PATH
    -pernode
    --bind-to none
    -x master=${master}
    -x NCCL_DEBUG
    -x PYTHONPATH
    -x NCCL_SOCKET_IFNAME=bond0
    -x NCCL_IB_GID_INDEX=3
    "
else
    # A卡配置分布式模式，主要是 NCCL_IB_HCA、NCCL_SOCKET_IFNAME 不同
    master=`head -1 ${TRAIN_WORKSPACE}/hostfile |awk '{print $1}'`
    mpirun="orterun
    --allow-run-as-root
    -tag-output
    -timestamp-output
    --hostfile ${TRAIN_WORKSPACE}/hostfile
    --mca btl_tcp_if_include $NCCL_SOCKET_IFNAME
    -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
    -x PATH
    -x LD_LIBRARY_PATH
    -pernode
    --bind-to none
    -x master=${master}
    -x NCCL_DEBUG
    -x PYTHONPATH
    -x NCCL_SOCKET_IFNAME=xgbe0 
    -x NCCL_IB_GID_INDEX=3 
    "
fi

# 执行数据准备脚本
${mpirun} sh prepare_data.sh
# ${mpirun} cp patch_code/utils.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/utils.py
# ${mpirun} rm -v /bin/python3
# ${mpirun} ln -s /bin/python3.12 /bin/python3
# TODO: change your own domain config file here
DOMAIN_CONFIG_FILE='generated_config.json'

BASE_CONFIG=(
    dsw      # Environment: dsw or dlc
    V59B    # Model Size
    3        # Micro Batch Size (per GPU)
    5760     # Global Batch Size (total across all GPUs)
    5e-5     # Max Learning Rate
    5e-5     # Min Learning Rate
    4096     # Sequence Length
    4096     # Padding Length
    bf16     # Precision    (fp16/bf16/fp8)
)


PARALLE_AND_BOOL_OPTION=(
    1        # TP: Tensor Parallelism
    4        # PP: Pipeline Parallelism
    1        # CP: Context Parallelism
    8        # EP: Expert Parallelism (for MoE models)
    true     # SP: Sequence Parallelism
    true     # DO: Distributed Optimizer
    true     # FL: Use Flash Attention (false uses Fused Attention)
    false    # SFT: Supervised Fine-Tuning mode

)

OTHERS=(
    sel      # AC: Activation Checkpointing (sel/full/none)
    false    # Optimizer Offload
    200   # Save Interval (in steps)
    "domain_config/${DOMAIN_CONFIG_FILE}" # Train Dataset Path
    "domain_config/${DOMAIN_CONFIG_FILE}" # Valid Dataset Path
    ./ckpt   # Load Checkpoint Path
    10000000000000    # Total Train Iterations (if SFT=true) or Tokens (if SFT=false)
    20000000000     # Warmup Iterations (if SFT=true) or Tokens (if SFT=false)
    ./ckpt/output # Output Base Path
    true        # NO_LOAD_OPTIM: 不加载优化器状态
    true        # NO_LOAD_RNG: 不加载随机数状态
    false       # NO_SAVE_OPTIM
    true        # 模型是否有MTP层
    local        # knowledge distillation (real/fake/local/none)
    127.0.0.1	 # teacher ip   支持多ip
    30000        # teacher port，如果是teacher type=local这个参数不会生效
    0.0         # lm-loss-weight
    1.0        # distill-loss-weight
    0.0        # mtp-loss-weight
    0.1        # mtp-distill-loss-weight
    0.0        # jsd-loss-weight
    0.9        # jsd-beta
    0.0       # mtp-jsd-loss-weight
    0.9        # mtp-jsd-beta
    "teacher"   # MP_DATASET_TYPE
    1.0       # temperature
    10000000000000      # eval-interval: 多少次迭代后，评估一次
    -1      # validation step 数: 正常情况>0, 设为-1时，valid data全过一遍
)

${mpirun} chmod +x ./projects/deepseek/run_mcore_deepseek.sh
export BINDPCIE_PATH=/root/paddlejob/bindpcie
${mpirun} -x BINDPCIE_PATH ./projects/deepseek/run_mcore_deepseek.sh  \
    ${BASE_CONFIG[@]} \
    ${PARALLE_AND_BOOL_OPTION[@]} \
    ${OTHERS[@]}

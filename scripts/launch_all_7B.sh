#!/bin/bash

WORKER_HOSTS=(
    "WORKER.IP.ADDRESS"
)
REMOTE_USER="root"
PROJECT_PATH="STAMP"
TRAIN_SCRIPT="train.main_uni"
SCRIPT_ARGS="" 

ENV_SETUP_CMDS="
export NCCL_IB_GID_INDEX=3;
export NCCL_IB_SL=3;
export NCCL_CHECKS_DISABLE=1;
export NCCL_P2P_DISABLE=0;
export NCCL_IB_DISABLE=0;
export NCCL_LL_THRESHOLD=16384;
export NCCL_IB_CUDA_SUPPORT=1;
export NCCL_SOCKET_IFNAME=bond1;
export UCX_NET_DEVICES=bond1;
export NCCL_IB_HCA='mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6';
export NCCL_COLLNET_ENABLE=0;
export SHARP_COLL_ENABLE_SAT=0;
export NCCL_NET_GDR_LEVEL=2;
export NCCL_IB_QPS_PER_CONNECTION=4;
export NCCL_IB_TC=160;
export NCCL_PXN_DISABLE=1;
"

if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project path '$PROJECT_PATH' does not exist."
    exit 1
fi

CONFIG_FILE="${PROJECT_PATH}/multi_node_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Accelerate config '$CONFIG_FILE' does not exist."
    exit 1
fi

pkill -f "accelerate"
for HOST in "${WORKER_HOSTS[@]}"; do
    ssh ${REMOTE_USER}@${HOST} "pkill -f 'accelerate'"
done
sleep 2

pip install -r ${PROJECT_PATH}/requirements.txt
for HOST in "${WORKER_HOSTS[@]}"; do
    ssh ${REMOTE_USER}@${HOST} "pip install -r ${PROJECT_PATH}/requirements.txt"
done
sleep 2

cd ${PROJECT_PATH} || exit

MASTER_CMD="${ENV_SETUP_CMDS} accelerate launch \
    --config_file ${CONFIG_FILE} \
    --machine_rank 0 \
    -m ${TRAIN_SCRIPT} ${SCRIPT_ARGS}"

eval "${MASTER_CMD}" &

RANK=1
for HOST in "${WORKER_HOSTS[@]}"; do
    REMOTE_CMD="cd ${PROJECT_PATH}; ${ENV_SETUP_CMDS} accelerate launch \
        --config_file ${CONFIG_FILE} \
        --machine_rank ${RANK} \
        -m ${TRAIN_SCRIPT} ${SCRIPT_ARGS}"

    ssh -n ${REMOTE_USER}@${HOST} "${REMOTE_CMD}" &

    RANK=$((RANK+1))
done

wait
echo "Training finished."
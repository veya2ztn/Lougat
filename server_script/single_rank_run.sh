#!/bin/bash
module load anaconda/2021.11
module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x



export WANDB_API_KEY=97f89f82c1c096c3ec08c67ee9abfc0b9c319960
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=https://zhangtianning.di:IDd1jJ7pW7MXd1od63GnWeASuzpqyx1lY8N3TESETAn62A8oOcQmLJHA7IyG@blsc-proxy.pjlab.org.cn:13128; done
export LOGGING_LEVEL=WARNING
export PYTHONUNBUFFERED=1
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# Using async gradient all reduce 
export CUDA_DEVICE_MAX_CONNECTIONS=1

### nodes gpus rank master_addr job_id
# nodes
NODES=$1
# gpus
NPROC_PER_NODE=$2

# rank
NODE_RANK=$3

# master
MASTER_ADDR=$4
MASTER_PORT="29501"

#JOB ID
BATCH_JOB_ID=$5

CMD=$6
# logs
echo "$NODE_RANK,$NODES,$NPROC_PER_NODE,$MASTER_ADDR,$BATCH_JOB_ID"
mkdir "log/${BATCH_JOB_ID}"
OUTPUT_LOG="log/${BATCH_JOB_ID}/train_rank${NODE_RANK}.log"


# export ACCELERATE_USE_DEEPSPEED=true #<----use this enable accelerate deepspeed
# export ACCELERATE_DEEPSPEED_CONFIG_FILE=config/deepspeed/deepspeed_config_s1_clip1.json #<----use this enable accelerate deepspeed
export ACCELERATE_USE_DEEPSPEED=false #<----use this enable accelerate deepspeed
export ACCELERATE_MIXED_PRECISION=bf16

# export ACCELERATE_DYNAMO_USE_DYNAMIC=true
# export ACCELERATE_ACCELERATE_DYNAMO_BACKEND=inductor
echo ${CMD}
# torchrun --nnodes="${NODES}"  --node_rank="${NODE_RANK}" --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
# train_via_accelerate.py --task train --model rlougat1KS --Dataset uparxive_parquet --root_name data \
#     --preload_weight checkpoints/UparxiveV2/Rlougat_1KS/rlougat1KS_1KS/best/weight/epoch0032/pytorch_model.bin \
#     --batch_size 16 --gradient_accumulation_steps 2 --lr 0.0005 \
#     --num_workers 8 --trial_name rlougat1KS_1KS_full \
#     --freeze_encoder False --start_count 2 --start_weight 2 --token_weight 1 --bbox_weight 1 --epochs 200 \
#     --use_wandb >> "${OUTPUT_LOG}" 

# torchrun --nnodes="${NODES}"  --node_rank="${NODE_RANK}" --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
# train_via_accelerate.py --task train --model flougatU_B2KS --Dataset uparxive_parquet2k --root_name data \
#     --preload_weight checkpoints/UparxiveV2.2k/FlougatU_B2KS/flougatU_B2KS/best/weight/epoch0020/pytorch_model.bin \
#     --batch_size 4 --gradient_accumulation_steps 5 --lr 0.0001 --load_weight_partial --load_weight_ignore_shape \
#     --num_workers 8 --coordinate_retreive_method mapping_coordinate_hard --trial_name flougatU_B2KS_full2 \
#     --freeze_decoder False --start_count 2 --start_weight 2 --token_weight 5 --bbox_weight 1 --epochs 100 \
#     --clip_value 0.1 --find_unused_parameters \
#     --use_wandb --wandbwatch 1000 >> "${OUTPUT_LOG}" 
torchrun --nnodes="${NODES}"  --node_rank="${NODE_RANK}" --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    train_via_accelerate.py --task train --model slougat_small --Dataset locr --root_name "" \
    --preload_weight pretrain_weights/slougat.matched_start.pt \
    --batch_size 10 --gradient_accumulation_steps 5 --lr 0.0001 --num_workers 8 \
    --coordinate_retreive_method mapping_coordinate_hard --trial_name slougat_small \
    --start_count 2 --start_weight 2 --token_weight 5 --bbox_weight 1 --find_unused_parameters \
    --epochs 200 --clip_value 1 --use_wandb --compile_image_encoder False >> "${OUTPUT_LOG}" 
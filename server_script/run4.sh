#!/bin/bash
#SBATCH -J Flougat      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-FlougatX.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-FlougatX.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH --job-name=FlougatX
#SBATCH --partition=AI4Phys
#SBATCH --nodes=8
#SBATCH --quotatype=spot
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
export WANDB_API_KEY=97f89f82c1c096c3ec08c67ee9abfc0b9c319960
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:IDd1jJ7pW7MXd1od63GnWeASuzpqyx1lY8N3TESETAn62A8oOcQmLJHA7IyG@10.1.20.50:23128; done
GPUS_PER_NODE=8
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=1 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

echo "WORLD_SIZE:" $WORLD_SIZE "NODE_RANK:" $NODE_RANK "DEVICES:" $CUDA_VISIBLE_DEVICES
echo "MASTER_ADDR:" $MASTER_ADDR "MASTER_PORT:" $MASTER_PORT "SLURM_PROCID:" $SLURM_PROCID
echo "NNODES": $NNODES

export CMD="""train_via_accelerate.py --task train --model flougatX_small \
--train_dataset_path data/arxiv_train_data/good/train_0306.jsonl \
--valid_dataset_path data/arxiv_train_data/good/validation_0306.jsonl --batch_size 8 \
--lr 0.00001 --use_wandb --num_workers 12 --random_dpi False \
--preload_state \
--preload_weight checkpoints/Lougat/Flougat_small/flougatX_small/best/weight/epoch0098/pytorch_model.bin \
--trial_name flougatX_small_more --coordinate_retreive_method mapping_coordinate_hard \
--start_count 1 --start_weight 10 --token_weight 1 --bbox_weight 5 --epochs 300 --save_every_epoch 10"""

export ACCELERATE_USE_DEEPSPEED=true #<----use this enable accelerate deepspeed
export ACCELERATE_DEEPSPEED_CONFIG_FILE=config/deepspeed/deepspeed_config_s1.json #<----use this enable accelerate deepspeed

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "
echo $LAUNCHER
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

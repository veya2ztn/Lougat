#!/bin/bash
#SBATCH -J Lougat             
#SBATCH -o log/%j-Lougat.out  
#SBATCH -e log/%j-Lougat.out  
#SBATCH --job-name=Lougat
#SBATCH --partition=vip_gpu_ailab_low
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --qos=gpugpu

### GPUS
GPUS=4
### 脚本名称
RANK_SCRIPT="single_rank_run.sh"
### Job Path
JOB_PATH=`pwd`
### Job ID
JOB_ID="${SLURM_JOB_ID}"
### 获取节点主机名
for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  rank[$k]=$(($k-1))
  echo ${host[$k]}
done

### 设置主节点,将第一个节点主机名做为 master 地址.
MASTER_ADDR=${host[1]}

### Nodes
NODES="${#host[@]}"

### nodes gpus rank master_addr job_id

bash ${RANK_SCRIPT} ${NODES} ${GPUS} 0 ${MASTER_ADDR} ${JOB_ID}&

for((i=2;i<=${NODES};i++));
do
   node_host=${host[$i]}
   node_rank=${rank[$i]}
   echo "nodes:${NODES}, host:${node_host}, node_rank:${node_rank}, master_addr:${MASTER_ADDR}"
   srun -N 1 --gres=gpu:$GPUS -w $node_host bash ${RANK_SCRIPT} ${NODES} ${GPUS} $node_rank ${MASTER_ADDR} ${JOB_ID}&  
done
wait

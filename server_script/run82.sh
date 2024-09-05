#!/bin/sh
#SBATCH -J STEADPreLoad      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-STEADPreLoad  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-STEADPreLoad  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286!@10.1.8.50:33128; done
accelerate-launch --config_file config/accelerate/bf16_single.yaml  train_single_station_via_llm_way.py -c checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/12_24_22_23_19-seed_21/train_config.json --preload_state --downstream_task mse_xymL4S_steadBO_unit --Resource STEAD --task train --lr 0.00001 --optim sophia --freeze_embedder --freeze_backbone --freeze_downstream --batch_size 12 --find_unused_parameters False --warmup_epochs 0 --epochs 100 --preload_weight checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/12_24_22_23_19-seed_21/pytorch_model.bin --use_wandb --max_length 12000 --status_type P-2to60 --use_confidence P-2to60 --resource_source stead.trace.BDLEELSSO.hdf5 --use_resource_buffer True --loader_all_data_in_memory_once True

# --NoiseGenerate pickalong_receive --noise_config_tracemapping_path datasets/STEAD/name2receivetype.json --noise_config_noise_namepool datasets/STEAD/stead.noise.csv --use_wandb True
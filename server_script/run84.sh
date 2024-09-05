#!/bin/sh
#SBATCH -J EarlyWarning      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-EarlyWarning  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-EarlyWarning  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286!@10.1.8.50:33128; done
accelerate-launch --config_file config/accelerate/fp32_cards8.yaml train_single_station_via_llm_way.py -c checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/STEAD.EarlyWarning/train_config.json --preload_state --downstream_task mse_xymL4S_steadBO_unit --Resource STEAD --task train --lr 0.0001 --optim sophia --freeze_embedder --freeze_backbone --freeze_downstream --batch_size 32 --find_unused_parameters False --warmup_epochs 0 --epochs 100 --preload_state checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/STEAD.EarlyWarning/checkpoints/checkpoint_99/ --preload_weight --use_wandb --max_length 6000 --status_type P-2to10 --use_confidence P-2to10 --resource_source stead.trace.BDLEELSSO.hdf5  --use_wandb True --trial_name STEAD.EarlyWarning.More 
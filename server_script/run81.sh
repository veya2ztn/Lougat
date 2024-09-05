#!/bin/sh
#SBATCH -J STEADBuffer      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-STEADBuffer  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-STEADBuffer  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286!@10.1.8.50:33128; done
accelerate-launch --config_file config/accelerate/bf16_cards8_ds1.yaml train_single_station_via_llm_way.py -c checkpoints/stead.trace.full.extend.addRevTypeNoise/Goldfish.40M_A.Sea/04_13_22_41-seed_21-284c/train_config.json  --Resource STEAD --task train --epochs 200 --preload_weight --use_resource_buffer True  --lr 0.0001 --scheduler none --normlize_for_stable 0 --retention_mode triton_parallel --NoiseGenerate pickalong_receive  --preload_weight checkpoints/stead.trace.full.extend.addRevTypeNoise/Goldfish.40M_A.Sea/04_13_22_41-seed_21-284c/pytorch_model.bin  --use_wandb

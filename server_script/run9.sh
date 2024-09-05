#!/bin/sh
#SBATCH -J DrawDiTing      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-DrawDiTing.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-DrawDiTing.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286@10.1.8.50:33128; done
export THEPATH=checkpoints/diting.group.full.good.hdf5/Goldfish.40M_A.Sea/goldfish40mditing2/visualize/DEV/ahead_L_to_the_sequence.w3000.l36000.c9000.Pad_zero_data/
python train_single_station_via_llm_way.py -c $THEPATH/infer_config.json --task infer_plot  --plot_data_dir $THEPATH  --Dataset ConCatDataset --upload_to_wandb True --Resource DiTing
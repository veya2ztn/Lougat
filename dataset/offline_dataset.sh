#!/bin/sh
#SBATCH -J OfflineData     # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/OfflineData.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/OfflineData.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH --job-name=OfflineData
#SBATCH --partition=AI4Phys
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpu=64

python dataset_offline_data_processing.py $1
#!/bin/bash
#SBATCH -J Lougat             
#SBATCH -o log/%j-Lougat.out  
#SBATCH -e log/%j-Lougat.out  
#SBATCH --job-name=Lougat
#SBATCH --partition=vip_gpu_ailab
#SBATCH --account=ai4earth
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
export datasetroot=data/archive_tex.colorful.csv.fold
export datasetpaths="$datasetroot/archive_tex.colorful.nobbl.perfect_successed.pdf_box_pair.csv \
$datasetroot/archive_tex.colorful.addbbl.perfect_successed.pdf_box_pair.csv
"
python -u train_via_accelerate.py --task train --model flougat_small --train_dataset_path $datasetpaths --root_name data --preload_state "" --preload_weight pretrain_weight/FlougatSmall_start.model.bin  --batch_size 14 --lr 0.0001 --num_workers 12 --random_dpi False --test_dataloader
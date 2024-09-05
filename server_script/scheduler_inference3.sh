#!/bin/sh
#SBATCH -J Inference      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-Inference.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-Inference.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286!@10.1.8.50:33128; done
for THEPATH in \
checkpoints/instance.group.all.hdf5/Goldfish.40M_A.Sea/03_05_19_42-seed_21-41fe \
# checkpoints/instance.group.HH.hdf5/Goldfish.40M_A.Sea/02_15_11_37-seed_21-3d59 \
# checkpoints/instance.group.HH.hdf5/Goldfish.40M_A.Sea/02_15_19_04-seed_21-2667 \
# checkpoints/instance.group.HH.hdf5/Goldfish.40M_A.Sea/02_15_20_45-seed_21-68e6;
do
    if [[ -e "$THEPATH" ]];then

        accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
        --Resource Instance --task recurrent_infer --valid_sampling_strategy.strategy_name early_warning_before_p --valid_sampling_strategy.early_warning 500 \
        --batch_size 16 --max_length 9000 --recurrent_chunk_size 9000 --recurrent_start_size 9000 --preload_state "" --upload_to_wandb True \
        --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" \
        --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False

        accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource Instance --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
        --Dataset ConCatDataset --component_concat_file datasets/INSTANCE/instance.valid.cat2series.list.npy --batch_size 6  \
        --max_length 36000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 6000 --recurrent_start_size 6000 --component_intervel_length 0 \
        --preload_state "" --upload_to_wandb True --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --status_type N0P1S2 --use_confidence whole_sequence \
        --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
        --padding_rule noise --padding_rule zero \
        --NoiseGenerate "nonoise" --noise_config_tracemapping_path "" --noise_config_noise_namepool ""
        
    else
        echo "Target path $THEPATH does not exist, skipping..."
    fi
done
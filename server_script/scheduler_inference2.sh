#!/bin/sh
#SBATCH -J Inference      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-Inference.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-Inference.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286!@10.1.8.50:33128; done
for THEPATH in \
checkpoints/stead.trace.BDLEELSSO.addRevTypeNoise/Goldfish.40M_A.Sea/STEAD.Noise.EarlyWarning;
do
    if [[ -e "$THEPATH" ]];then

        accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
        --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name early_warning_before_p --valid_sampling_strategy.early_warning 500 \
        --batch_size 32 --max_length 18000 --recurrent_chunk_size 9000 --recurrent_start_size 9000 --preload_state "" --upload_to_wandb True \
        --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --NoiseGenerate pickalong_receive \
        --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False



        # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
        # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6  \
        # --max_length 24000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 3000 --recurrent_start_size 3000 --component_intervel_length 9000 \
        # --preload_state "" --upload_to_wandb True --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --status_type N0P1S2 --use_confidence whole_sequence \
        # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
        # --resource_source stead.trace.BDLEELSSO.hdf5 --padding_rule noise --padding_rule noise \
        # --NoiseGenerate pickalong_receive --noise_config_tracemapping_path datasets/STEAD/name2receivetype.json --noise_config_noise_namepool datasets/STEAD/stead.noise.csv

        # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
        # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6  \
        # --max_length 24000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 3000 --recurrent_start_size 3000 --component_intervel_length 6000 \
        # --preload_state "" --upload_to_wandb True --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --status_type N0P1S2 --use_confidence whole_sequence \
        # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
        # --resource_source stead.trace.BDLEELSSO.hdf5 --padding_rule noise --padding_rule noise \
        # --NoiseGenerate pickalong_receive --noise_config_tracemapping_path datasets/STEAD/name2receivetype.json --noise_config_noise_namepool datasets/STEAD/stead.noise.csv

        # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
        # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6  \
        # --max_length 24000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 3000 --recurrent_start_size 3000 --component_intervel_length 9000 \
        # --preload_state "" --upload_to_wandb True --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --status_type N0P1S2 --use_confidence whole_sequence \
        # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
        # --resource_source stead.trace.BDLEELSSO.hdf5 --padding_rule noise --padding_rule zero \
        # --NoiseGenerate "nonoise" --noise_config_tracemapping_path "" --noise_config_noise_namepool ""

    else
        echo "Target path $THEPATH does not exist, skipping..."
    fi
done
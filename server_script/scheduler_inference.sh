#!/bin/sh
#SBATCH -J Inference      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-Inference.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-Inference.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286!@10.1.8.50:33128; done
for THEPATH in \
checkpoints/diting.group.good.hdf5/Pearlfish.40M_A.Sea/03_16_12_12-seed_42-0baf ;
do
    if [[ -e "$THEPATH" ]];then
        accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
        --Resource DiTing --task recurrent_infer --valid_sampling_strategy.strategy_name early_warning_before_p \
        --valid_sampling_strategy.early_warning 500 \
        --batch_size 32 --max_length 9000 --recurrent_chunk_size 9000 --preload_state "" --upload_to_wandb True \
        --freeze_embedder "" --freeze_backbone "" --freeze_downstream ""  \
        --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False

        # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
        # --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name early_warning_before_p --valid_sampling_strategy.early_warning 500 \
        # --batch_size 16 --max_length 9000 --recurrent_chunk_size 9000 --recurrent_start_size 9000 --preload_state "" --upload_to_wandb True \
        # --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --NoiseGenerate pickalong_receive \
        # --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False

        # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource DiTing --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
        # --Dataset ConCatDataset --component_concat_file datasets/DiTing330km/ditinggroup.full.subcluster.valid.cat2series.list.npy --batch_size 6  \
        # --max_length 36000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 3000 --recurrent_start_size 3000 --component_intervel_length 9000 \
        # --preload_state "" --upload_to_wandb True --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --status_type N0P1S2 --use_confidence whole_sequence \
        # --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --padding_rule interpolation 

        # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
        # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource DiTing --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
        # --Dataset ConCatDataset --component_concat_file datasets/DiTing330km/ditinggroup.full.subcluster.valid.cat2series.list.npy --batch_size 6  \
        # --max_length 36000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 3000 --recurrent_start_size 3000 --component_intervel_length 9000 \
        # --preload_state "" --upload_to_wandb True --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --status_type N0P1S2 --use_confidence whole_sequence \
        # --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --padding_rule zero 

    else
        echo "Target path $THEPATH does not exist, skipping..."
    fi
done


from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from .optimizer.optimizer_arguements import OptimizerConfig
from .scheduler.scheduler_arguements import SchedulerConfig,TimmCosineConfig, NoSchedulerConfig, TransformersCosineConfig
from model.model_arguements import FreezeConfig
from simple_parsing.helpers.serialization import to_dict
###############################################
############## Dataset Config  ################
###############################################


@dataclass
class MonitorConfig:
    use_wandb: bool = field(default=False)
    wandbwatch: int = field(default=0)
    log_interval: int = field(default=50)

import os
@dataclass
class CheckpointConfig:
    num_max_checkpoints: int = field(default=1)
    save_every_epoch: int = field(default=1)
    save_warm_up: int = field(default=5)
    preload_weight: Optional[str] = None
    preload_state: Optional[str] = None
    load_weight_partial: bool = field(default=False)
    load_weight_ignore_shape: bool = field(default=False)
    continue_train: bool = field(default=False)
    preload_dir: Optional[str] = None
    def __post_init__(self):
        if self.preload_dir is not None and self.preload_state is None:
            checkpoint_order   = os.listdir(self.preload_dir)[0]
            self.preload_state = os.path.join(self.preload_dir, checkpoint_order)
            self.continue_train= True
@dataclass
class TaskConfig:
    pass

@dataclass
class TrainConfig(TaskConfig):
    Optimizer: OptimizerConfig
    Monitor: MonitorConfig
    Checkpoint: CheckpointConfig
    Freeze      : FreezeConfig
    Scheduler: SchedulerConfig = subgroups(
        {
            "TimmCosine": TimmCosineConfig,
            "none": NoSchedulerConfig,
            "TFCosine": TransformersCosineConfig,
        },
        default='TimmCosine'
    )
    
    clean_checkpoints_at_end: bool = field(default=False)
    epochs: int = field(default=100)
    seed: int = field(default=42)
    gradient_accumulation_steps : int = field(default=1)
    clip_value: Optional[float] = None
    not_valid_during_train : bool = field(default=False)
    lr : float = field(default=1e-5)
    find_unused_parameters: bool = field(default=False)
    save_on_epoch_end: bool = field(default=True)
    train_or_infer: str = field(default='train')
    do_validation_at_first_epoch: bool = field(default=False)
    
    time_test: bool = field(default=False)
    preduce_meanlevel: str = field(default=None)
    def __post_init__(self):
        self.Optimizer.lr = self.lr
        self.Scheduler.lr = self.lr
        self.Scheduler.epochs = self.epochs
        self.train_or_infer = 'train'

def get_parallel_config_of_accelerator(accelerator):
    parallel_config = {}
    for key, val in accelerator.state._shared_state.items():
        if isinstance(val, (list, tuple, str, int, bool)) or val is None:
            parallel_config[key] = val
        elif key in ['deepspeed_plugin']:
            parallel_config[key] = to_dict(val)
            parallel_config[key]['hf_ds_config'] = parallel_config[key]['hf_ds_config'].config
        else:
            accelerator.print(f"skip unserializable key {key}={val}")
    return parallel_config
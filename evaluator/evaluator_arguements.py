from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from train.train_arguements import TaskConfig, CheckpointConfig
from pathlib import Path
import os, logging, sys

@dataclass
class EvaluatorConfig(TaskConfig):
    Checkpoint       : CheckpointConfig
    infer_mode       : str = None
    upload_to_wandb  : bool = False
    clean_up_plotdata: bool = False
    train_or_infer : str = 'infer'
    eval_dataflag: str = 'DEV'
    save_monitor_num: int = 10
    do_fasttest_eval: bool = False

    def __post_init__(self):
        self.train_or_infer   = 'infer'
        raise NotImplementedError('we have not implement it yet')
        

@dataclass
class EvalPlotConfig(TaskConfig):
    plot_data_dir: str 
    infer_mode    : str = None
    upload_to_wandb: bool = False
    train_or_infer : str = 'infer_plot'
    do_fasttest_eval: bool = False
    def __post_init__(self):
        assert self.infer_mode is not None,  "infer_mode must be specified"
        self.train_or_infer = 'infer_plot'
        raise NotImplementedError('we have not implement it yet')
    

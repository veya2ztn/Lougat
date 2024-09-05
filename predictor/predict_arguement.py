from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from train.train_arguements import TaskConfig, CheckpointConfig
from pathlib import Path
import os, logging, sys

@dataclass
class PredictorConfig(TaskConfig):
    pretrain_path   : str = field(help="Path to the pretrain model.")
    input_path      : str|List[str]|Tuple[str] = field(help="PDF(s) to process.")
    output_path     : str
    batchsize       : int  = 1
    return_attention: bool = False
    recompute       : bool = False
    markdown        : bool = False
    random_seed     : int  = 46
    
    def __post_init__(self):
        if not os.path.exists(self.output_path):
            raise NotImplementedError("Output directory does not exist. Creating output directory.")
        if self.return_attention:   # return attention page by page
            logging.info("Warning: in return_attention mode, we force set batchsize=1")
            self.batchsize = 1
            self.recompute = True
        


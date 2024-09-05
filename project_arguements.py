from simple_parsing import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from simple_parsing import NestedMode
from argparse import Namespace
from train.train_arguements import TrainConfig, TaskConfig
from config.utils import print_namespace_tree, dict_to_arglist
from simple_parsing.helpers.serialization import to_dict,save
from utils import print0
import accelerate
import json
from evaluator.evaluator_arguements import EvaluatorConfig,EvalPlotConfig
from dataset.dataset_arguements import DataloaderConfig
from model.model_arguements import (PromptNougatModelConfig, 
                                    PromptNougatBaseConfig, PromptNougatSmallConfig, 
                                    LlougatBaseConfig, LlougatSmallConfig, 
                                    FlougatSmallConfig, FlougatBaseConfig,FlougatXSmallConfig,UpArouGatBetaSConfig,UpArouGatBeta2KSConfig,
                                    UpArouGatSmallConfig,UpArouGatXSmallConfig,UpArouGat2KSConfig,UpArouGat1KSConfig,
                                    SlougatSmallConfig, SlougatBaseConfig,
                                    RlougatSmallConfig,Rlougat1KSConfig
                                    )

@dataclass
class ProjectConfig:
    
    DataLoader: DataloaderConfig
    task: TaskConfig = subgroups(
        {
            "train": TrainConfig,
            # "infer": EvaluatorConfig,
            # "infer_plot": EvalPlotConfig
        }
    )
    model:  PromptNougatModelConfig = subgroups(
        {'lougat_small':PromptNougatSmallConfig, 
         'lougat_base': PromptNougatBaseConfig,
         'llougat_small': LlougatSmallConfig,
         'llougat_base': LlougatBaseConfig,
         'flougat_small': FlougatSmallConfig,
         'flougatX_small':FlougatXSmallConfig,
         'flougatU_beta_1KS':UpArouGatBetaSConfig,
         'flougatU_1KS':UpArouGat1KSConfig,
         
         'flougatU_B2KS':UpArouGatBeta2KSConfig,
         'flougatU_small':UpArouGatSmallConfig,
         'flougatU2KS':UpArouGat2KSConfig,
         'flougatU_Xsmall':UpArouGatXSmallConfig,
         'flougat_base':  FlougatBaseConfig,
         'slougat_small':  SlougatSmallConfig,
         'slougat_base':  SlougatBaseConfig,
         'rlougat_small':  RlougatSmallConfig,
         'rlougat1KS':  Rlougat1KSConfig,

         },
        default='lougat_small'
        )
    output_dir: str = field(default=None)
    trial_name: str = field(default='')
    accelerate_version: str = field(default='')
    quick_debug:bool= False
    quick_test_inference:bool = False
    test_dataloader:bool= False
    
    def get_name(self):
        raise NotImplementedError
        shape_string = "_".join( [str(self.history_length)] + [str(t) for t in self.img_size])
        return f'{self.model_type}.{shape_string}.{self.in_chans}.{self.out_chans}.{self.embed_dim}.{self.depth}.{self.num_heads}'

    def __post_init__(self):
        output_dir, dataset_name, model_name, Name =  get_the_correct_output_dir(self)
        self.output_dir   = output_dir
        self.dataset_name = dataset_name
        self.model_name   = model_name
        self.trial_name   = Name
        self.accelerate_version = accelerate.__version__
        ### share the global keys
        self.DataLoader.Dataset._max_length             = self.model.decoder.max_length
        self.DataLoader.Dataset._input_size             = self.model.encoder.input_size
        self.DataLoader.Dataset._processor              = self.model.processor
        self.DataLoader.Dataset._decoder_start_token_id = self.model.decoder.decoder_start_token_id


import os,re
import time,hashlib
def get_the_correct_output_dir(args: ProjectConfig):
    model_name   = args.model.model_config_name
    dataset_name = args.DataLoader.Dataset.name
    if isinstance(args.task, EvalPlotConfig):
        output_dir= args.task.plot_data_dir
        dataset_name_from_path = re.findall(r"checkpoints\/(.*?)\/", args.task.plot_data_dir)[0]
        if dataset_name_from_path != dataset_name:
            print0(f"Warning: the dataset_name {dataset_name} is not equal to the dataset_name in the plot_data_dir {dataset_name_from_path}. We will use the {dataset_name} assigned")
            #dataset_name = dataset_name_from_path
        path_in_list = output_dir.split('/')
        position_of_datasetname = path_in_list.index(dataset_name_from_path)
        model_name = path_in_list[position_of_datasetname+1]
        Name = path_in_list[position_of_datasetname+2]
        output_dir = '/'.join(path_in_list[:position_of_datasetname+3])
        
    elif (isinstance(args.task, TrainConfig) and args.task.Checkpoint.continue_train) or isinstance(args.task, EvaluatorConfig):
        assert args.task.Checkpoint.preload_state or args.task.Checkpoint.preload_weight
        if args.task.Checkpoint.preload_state:
            output_dir = os.path.dirname(os.path.dirname(args.task.Checkpoint.preload_state))
        elif args.task.Checkpoint.preload_weight:
            output_dir = os.path.dirname(args.task.Checkpoint.preload_weight)
        
        Name = [part for part in output_dir.split("/") if '-seed_' in part ]
        if len(Name)==1:
            Name= Name[0]
        else:
            Name = output_dir.strip("/").split("/")[-1]
        if isinstance(args.task, EvaluatorConfig):
            dataset_name_from_path = re.findall(r"checkpoints\/(.*?)\/", output_dir)[0]
            if dataset_name_from_path != dataset_name:
                print0(f"Warning: the dataset_name {dataset_name} is not equal to the dataset_name in the plot_data_dir {dataset_name_from_path}. We will use the dataset_name assigned")
    else:
        if args.task.Checkpoint.preload_state and not args.task.Checkpoint.continue_train:
            print0("we don't continue train. We just start from a pretrained weight ")
        if args.trial_name:
            Name = args.trial_name
        else:
            # Name = time.strftime("%m_%d_%H_%M_%S") ###<-- there is a bug here. when multinode training
            rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
            local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
            rank = rank + local_rank
            confighash = hashlib.md5(str(args).encode("utf-8")).hexdigest()
            cachefile = "log/records"
            recordfile = os.path.join(cachefile, confighash)
            if rank == 0:
                if not os.path.exists(cachefile):os.makedirs(cachefile)
                Name = time.strftime("%m_%d_%H_%M")
                Name = f"{Name}-seed_{args.task.seed}-{confighash[:4]}"
                with open(recordfile, "w") as f:
                    json.dump({'Name':Name}, f)
            else:
                time.sleep(1)
                time_start = time.time()
                while not os.path.exists(recordfile):
                    time_cost =  time.time() - time_start
                    if time_cost > 60: raise TimeoutError(f"rank {rank} can't find the recordfile {recordfile} in 10s")
                with open(recordfile, "r") as f:
                    Name = json.load(f)['Name']
            #print(f"GPU:{rank}:Name:{Name}")
        output_dir = f'checkpoints/{dataset_name}/{model_name}/{Name}'
        
    dataset_name = dataset_name.split('_')[0] # remove all the subfix like _sample6000 
    return output_dir, dataset_name, model_name, Name



# def get_args(config_path=None):
#     args = simple_parsing.parse(config_class=ProjectConfig, config_path=None, args=None, add_config_path_arg=True)
#     return args


def get_args(config_path=None, args=None):

    conf_parser = ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--conf_file",  default=None, help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args(args)
    config_path = config_path if config_path else args.conf_file
    defaults = {}
    if config_path:
        with open(config_path, 'r') as f:defaults = json.load(f)
        defaults['use_wandb'] = False
    the_old_version_config = config_path is not None and 'DataLoader' not in defaults
    if the_old_version_config:
        form_args =  dict_to_arglist(defaults)
        whole_args= form_args + remaining_argv # put the remaining argv at the end, thus that can overwrite the default config
        
        parser = ArgumentParser()
        parser.add_arguments(ProjectConfig, dest="config")
        args,remaining_argv = parser.parse_known_args(whole_args)
        if len(remaining_argv) > 0:
            print0("Warning: some arguements are not parsed: ", remaining_argv)
            if 'train' in args:
                assert '--freeze_mode' not in remaining_argv, "--freeze_mode is deprecated. Please use task.Freeze(freeze_embedder=?,freeze_backbone=?,freeze_downstream=?) instead"
        return args.config
    else:
        assert config_path is None, "the new version config file is not supported yet"
        parser = ArgumentParser(nested_mode=NestedMode.WITHOUT_ROOT,config_path=config_path)
        parser.add_arguments(ProjectConfig, dest="config")
        args = parser.parse_args(remaining_argv)
        return args.config
    
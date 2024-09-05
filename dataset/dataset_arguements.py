from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from model.model_arguements import ProcessorConfig

###############################################
############## Dataset Config  ################
###############################################
@dataclass
class DatasetConfig:
    debug: bool = field(default=False)
    dataset_name : str = field(default=None)
    load_in_all_processing: bool = field(default=False)
    return_trend: bool = field(default=False)   
    dataset_version: str = field(default='alpha')
    @property
    def name(self):
        if self.debug:return "debug"
        raise NotImplementedError



@dataclass
class NougatDatasetConfig(DatasetConfig):
    train_dataset_path: List[str] = field(default_factory=None)
    valid_dataset_path: List[str] = field(default_factory=None)
    prompt_label_length:int = 64
    root_name: str = ""
    align_long_axis: bool = None
    random_dpi: bool = False
    random_noise_scale: float = 0.5
    _max_length              = None # make sure same as model max_length
    _input_size              = None
    _decoder_start_token_id  = None
    _processor               = None
    
    @property
    def name(self)->str:
        if self.debug:return "debug"
        raise NotImplementedError("Please assign dataset name ")
        return "Lougat"

    @property
    def max_length(self)->int:
        assert self._max_length is not None, "you need align the dataset max length same as model max length"
        return self._max_length
    
    @property
    def processor(self)->ProcessorConfig:
        assert self._processor is not None, "you need inherit processor from model config"
        return self._processor

    @property
    def input_size(self)->int:
        assert self._input_size is not None, "you need align the dataset input_size same as model input_size"
        return self._input_size


    @property
    def decoder_start_token_id(self)->int:
        assert self._decoder_start_token_id is not None,  "you need align the dataset decoder_start_token_id same as model decoder_start_token_id"
        return self._decoder_start_token_id

    def __post_init__(self):
        
        if self.valid_dataset_path is None:
            self.valid_dataset_path = self.train_dataset_path
        # if len(self.train_dataset_path)==1:
        #     self.train_dataset_path = self.train_dataset_path[0]
        # if len(self.valid_dataset_path)==1:
        #     self.valid_dataset_path = self.valid_dataset_path[0]

@dataclass
class UparxiveDatasetConfig(NougatDatasetConfig):
    train_dataset_path: List[str] = field(default_factory=["data/archive_tex.colorful.csv.fold/archive_tex.colorful.addbbl.perfect_successed.pdf_box_pair.csv",
                                                        #    "data/archive_tex.colorful.csv.fold/archive_tex.colorful.nobbl.perfect_successed.pdf_box_pair.csv",
                                                        #    "data/archive_tex.colorful.csv.fold/archive_tex.colorful.addbbl.partial_successed.pdf_box_pair.csv",
                                                        #    "data/archive_tex.colorful.csv.fold/archive_tex.colorful.nobbl.partial_successed.pdf_box_pair.csv"
                                                           ].copy)
    valid_dataset_path: List[str] = field(default=None)
    @property
    def name(self)->str:
        if self.debug:return "debug"
        return "Uparxive"
    

@dataclass
class ParquetUpaxiveConfig(NougatDatasetConfig):
    train_dataset_path: List[str] = field(default="data/uparxive_bbox_pair_parquet_V2_2k/Train")
    valid_dataset_path: List[str] = field(default="data/uparxive_bbox_pair_parquet_V2_2k/Valid")
    @property
    def name(self)->str:
        if self.debug:return "debug"
        return "UparxiveV2"

@dataclass
class ParquetUpaxive2kConfig(ParquetUpaxiveConfig):
    train_dataset_path: List[str] = field(default="data/uparxive_bbox_pair_parquet_V2_2k/Train")
    valid_dataset_path: List[str] = field(default="data/uparxive_bbox_pair_parquet_V2_2k/Valid")
    @property
    def name(self)->str:
        if self.debug:return "debug"
        return "UparxiveV2.2k"


@dataclass
class LocrDatasetConfig(NougatDatasetConfig):
    train_dataset_path: List[str] = 'data/arxiv_train_data/good/train_0306.jsonl'
    valid_dataset_path: List[str] = 'data/arxiv_train_data/good/validation_0306.jsonl'
    @property
    def name(self)->str:
        if self.debug:return "debug"
        return "Locr"
    
###############################################
############## Dataloader Config  #############
###############################################
@dataclass
class DataloaderConfig:
    Dataset : NougatDatasetConfig=subgroups({
        "locr": LocrDatasetConfig,
        "uparxive": UparxiveDatasetConfig,
        "uparxive_parquet": ParquetUpaxiveConfig,
        "uparxive_parquet2k":ParquetUpaxive2kConfig
    })
    shuffle : bool = field(default=True)
    num_workers : int= field(default=0)
    batch_size : int = field(default=2)
    data_parallel_dispatch: bool = field(default=True)
    donot_use_accelerate_dataloader: bool = field(default=False)
    not_pin_memory: bool = field(default=False)
    loader_all_data_in_memory_once: bool = field(default=False)
    
    # def __post_init__(self):
    #     self.Dataset.load_in_all_processing = self.data_parallel_dispatch
    #     self.Dataset.Resource.load_in_all_processing = self.data_parallel_dispatch



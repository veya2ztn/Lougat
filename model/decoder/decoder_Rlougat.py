from dataclasses import dataclass,fields,field
from .Rlougat.configuration_Rlougat import RlougatConfig, RlougatVEDConfig
from typing import Optional, List, Union


@dataclass
class RlougatDecoderConfig(RlougatConfig):
    
    max_position_embeddings:int = 4096
    max_length:int              = 4096
    hidden_size:int             = 768
    intermediate_size: int      = 2048
    num_hidden_layers: int      = 4
    num_attention_heads: int    = 16
    decoder_start_token_id: int = 0
    attn_implementation: str    = "flash_attention_2" #"eager" #
    coordinate_retreive_method: str = "position_revert"
    use_start_pos_embedding: bool = False
    pos_resolution: int = 2000

    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    def huggingface_dict(self):
        return {'model_type': self.model_type} | self.to_fields_dict()

    ## WARNING!!! YOUR CAN NOT OVERRIDE THE to_dict method~! use other method
    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
@dataclass
class SmallRlougatConfig(RlougatDecoderConfig):
    num_hidden_layers: int = field(default=4)
    
@dataclass
class BaseRlougatConfig(RlougatDecoderConfig):
    num_hidden_layers: int = field(default=10)

@dataclass
class MidRlougatConfig(RlougatDecoderConfig):
    hidden_size:int             = 1024
    intermediate_size: int      = 1024
    num_hidden_layers: int      = 4
    num_attention_heads: int    = 8
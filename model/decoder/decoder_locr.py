from dataclasses import dataclass,fields,field
from .locr.configuration_locr import PromptBartConfig, LougatConfig
from typing import Optional, List, Union



@dataclass
class PromptDecoderConfig(PromptBartConfig):
    image_embedding_size: List[int] = field(default_factory=lambda: [28,21])
    max_position_embeddings:int = 4096
    max_length:          int    = 4096
    decoder_layers:int = 12
    decoder_ffn_dim:int= 4096
    decoder_attention_heads:int= 16
    d_model: int = 1024
    position_decoder_hidden_size:int = 256
    position_decoder_layers:int = 3
    position_decoder_decay:float = 1
    add_cross_attention:bool = True
    scale_embedding: bool = True
    add_final_layer_norm:bool =True
    prompt_embed_dim:int=1024#,256,
    decoder_start_token_id:Optional[int] = 0
    forced_eos_token_id:Optional[int] = 2
    hidden_dimension: int = 1024
    omit_ratio: float = 0
    BartAttention_implement: str = 'sdpa'
    use_image_bias: bool = False
    #_use_flash_attention_2=False


    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    def huggingface_dict(self):
        return {'model_type':self.model_type}|self.to_fields_dict()

    ## WARNING!!! YOUR CAN NOT OVERRIDE THE to_dict method~! use other method
    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
@dataclass
class SmallLOCRConfig(PromptDecoderConfig):
    max_position_embeddings: int = field(default=4096)
    decoder_layers: int = field(default=4)
    max_length: int = field(default=4096)
@dataclass
class BaseLOCRConfig(PromptDecoderConfig):
    decoder_layers: int = field(default=10)
    max_length: int = field(default=4096)
    max_position_embeddings: int = field(default=4096)
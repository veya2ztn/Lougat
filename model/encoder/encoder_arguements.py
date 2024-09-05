from dataclasses import dataclass,fields,field
from typing import Optional, Union, List,Tuple
import os
from ..processor_arguements import ProcessorConfig
from transformers import SwinConfig, SwinModel
from .florence.configuration_florence2 import Florence2VisionConfig

from .sam2.sam_encoder import SamFeatureConfig

@dataclass
class FlorenceConfig(Florence2VisionConfig):
    
    input_size: List[int]  
    image_embedding_size: List[int]
    patch_size: List[int]
    patch_stride: List[int]
    patch_padding: List[int]
    dim_embed: List[int]
    num_heads: List[int]
    num_groups: List[int]
    depths: List[int]
    patch_prenorm: List[bool]
    window_size: int =  12
    projection_dim: int = 768

    _tokenizer = None

    @property
    def processor(self) -> ProcessorConfig:
        assert hasattr(self, "_tokenizer"), "you need inherit processor from model config"
        return self._processor

    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def huggingface_dict(self):
        return {'image_size': self.input_size,'model_type': self.model_type} | self.to_fields_dict()

    

    @property
    def image_size(self):
        return self.input_size


@dataclass
class NormalFlorenceConfig(FlorenceConfig):
    input_size   : List[int] = field(default_factory=lambda: [768, 768])
    image_embedding_size: List[int] = field(default_factory=lambda:[24,24])
    patch_size   : List[int]=field(default_factory=lambda:[7, 3, 3, 3])
    patch_stride : List[int]=field(default_factory=lambda:[4, 2, 2, 2])
    patch_padding: List[int]=field(default_factory=lambda:[3, 1, 1, 1])
    dim_embed    : List[int]=field(default_factory=lambda:[128, 256, 512, 1024])
    num_heads    : List[int]=field(default_factory=lambda:[4, 8, 16, 32])
    num_groups   : List[int]=field(default_factory=lambda:[4, 8, 16, 32])
    depths       : List[int]=field(default_factory=lambda:[1, 1, 9, 1])
    patch_prenorm: List[bool]=field(default_factory=lambda: [False, True, True, True])
    window_size   :int =   field(default=12)
    projection_dim:int =   field(default=768)

@dataclass
class Florence1KConfig(NormalFlorenceConfig):
    input_size   : List[int]        = field(default_factory=lambda: [1024, 768])
    image_embedding_size: List[int] = field(default_factory=lambda: [  32,  24])



@dataclass
class Florence2KConfig(NormalFlorenceConfig):
    input_size   : List[int]        = field(default_factory=lambda: [2048, 1472])
    image_embedding_size: List[int] = field(default_factory=lambda:[32,23])
    patch_size   : List[int]=field(default_factory=lambda:[7, 7, 3, 3])
    patch_stride : List[int]=field(default_factory=lambda:[4, 4, 2, 2])
    patch_padding: List[int]=field(default_factory=lambda:[3, 3, 1, 1])


@dataclass
class FlorenceXConfig(NormalFlorenceConfig):
    input_size   : List[int] = field(default_factory=lambda: [1024, 1024])
    image_embedding_size: List[int] = field(default_factory=lambda:[32,32])

@dataclass
class SwinEncoderConfig(SwinConfig):
    
    input_size: List[int]        #help="Input image size (width, height)"
    image_embedding_size: List[int]
    window_size: int             #help="Window size(=patch size) of SwinTransformer"
    encoder_layer: List[int]     #help="Number of layers of SwinTransformer encoder"
    patch_size: int              #help="Patch size of SwinTransformer"
    embed_dim: int               #help="Embedding dimension of SwinTransformer"
    num_heads: List[int]         #help="Number of heads of SwinTransformer"
    #name_or_path: Union[str, bytes, os.PathLike] = field(default=None) #'swin_base_patch4_window12_384'
    num_channels: int = 3 ##

    _tokenizer = None ## 
    
    @property
    def processor(self)->ProcessorConfig:
        assert hasattr(self, "_tokenizer"), "you need inherit processor from model config"
        return self._processor
    
    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    def huggingface_dict(self):
        return {'image_size':self.input_size, 'depths':self.encoder_layer,
                'model_type':self.model_type,
                '_out_features':None,
                '_out_indices':None,
                }|self.to_fields_dict()
    def __post_init__(self):  
        super().__init__(image_size=self.input_size, depths=self.encoder_layer, **self.__dict__ )

@dataclass
class NormalSwinConfig(SwinEncoderConfig):
    input_size: List[int] = field(default_factory=lambda: [896, 672])
    image_embedding_size: List[int] = field(default_factory=lambda: [28, 21])
    encoder_layer: List[int] = field(default_factory=lambda: [2, 2, 14, 2])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    window_size: int = field(default=7)
    patch_size: int = field(default=4)
    embed_dim: int = field(default=128)
    
    

@dataclass
class SamEncoderConfigBase(SamFeatureConfig):
    
    image_size: List[int]  
    image_embedding_size: List[int]  
    encoder_embedding_size: int = 768  
    compile_image_encoder: bool = True  
    _tokenizer = None ## 
    
    @property
    def input_size(self):
        return [self.image_size,self.image_size]
    
    @property
    def processor(self)->ProcessorConfig:
        assert hasattr(self, "_tokenizer"), "you need inherit processor from model config"
        return self._processor
    
    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
        
    def huggingface_dict(self):
        return {'model_type':self.model_type,'hidden_size':self.encoder_embedding_size}|self.to_fields_dict()

@dataclass
class NormalSamEncoderConfig(SamEncoderConfigBase):
    image_size: List[int] = field(default_factory=lambda: [1024, 768])
    image_embedding_size: List[int] = field(default_factory=lambda:[32,24]) ### use 
    encoder_embedding_size: int = 768
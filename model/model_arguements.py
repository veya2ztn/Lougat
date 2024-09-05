from dataclasses import dataclass,fields
from typing import Optional, Union, List,Tuple
import os,json

from .encoder.encoder_arguements import (SwinEncoderConfig,Florence2KConfig,  NormalSwinConfig, 
                                         FlorenceConfig, NormalFlorenceConfig, Florence1KConfig,
                                         SamEncoderConfigBase, NormalSamEncoderConfig,FlorenceXConfig)
from .processor_arguements import (ProcessorConfig, NougatProcessorConfig, FlougatProcessorConfig, Uparxive1KProcessorConfig,
                                   UparxiveProcessor2kConfig,LougatProcessorConfig, SlougatProcessorConfig, UparxiveBetaProcessorConfig,UparxiveBetaProcessor2kConfig,
                                   FlougatXProcessorConfig, UparxiveProcessorConfig,UparxiveXProcessorConfig)

from .decoder.decoder_locr import PromptDecoderConfig, BaseLOCRConfig, SmallLOCRConfig, LougatConfig
from .decoder.decoder_llougat import LlougatDecoderConfig, SmallLlougatConfig, BaseLlougatConfig, LlougatVEDConfig
from .decoder.decoder_Rlougat import RlougatDecoderConfig, RlougatVEDConfig, BaseRlougatConfig, SmallRlougatConfig, MidRlougatConfig
from transformers import VisionEncoderDecoderConfig


@dataclass
class LossConfig:
    start_count : int  = 5
    token_weight: float= 5
    bbox_weight : float= 1
    start_weight: float= 1
    full_weight : float= 1
@dataclass
class TrainerModelConfig(VisionEncoderDecoderConfig):
    loss_config:LossConfig

    @property
    def name(self):
        if self.model_config_name is None:
            raise NotImplementedError
        return self.model_config_name

    @property
    def nickname(self):
        if self.model_config_name is None:
            raise NotImplementedError
        return self.model_config_name
    
    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Overrite the to_json_file method to remove the _processor attr
        """
        import copy
        self = copy.deepcopy(self)
        ## remove the _processor attr
        dirpath = os.path.dirname(json_file_path)
        self.processor.get_processor().save_pretrained(dirpath) # <--- save the processor
        del self.processor
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))
        

    def get(self, key, default=None):
        pool = vars(self)
        if key in pool:return pool[key]
        return default
    
    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class PromptNougatModelConfig(TrainerModelConfig):
    ## below is only for simple parsing tracking
    encoder: SwinEncoderConfig
    decoder: PromptDecoderConfig
    processor: ProcessorConfig

    def huggingface_dict(self):
        #assert self.model_config_name is not None, "Since the model_config_name is None, it seem you are not assign any model by --model [model] "
        #super().__init__(
        return LougatConfig(
            encoder = self.encoder.huggingface_dict(),
            decoder = self.decoder.huggingface_dict() | {'image_size': self.encoder.image_size, 'image_embedding_size': self.encoder.image_embedding_size,
                                                       'vocab_size':len(self.processor.get_processor().tokenizer)}
        )
    
    @staticmethod
    def from_pretrained(path,**kargs):
        path = os.path.join(path,'config.json')
        with open(path,'r') as f: config = json.load(f)
        encoder   = SwinEncoderConfig(**{field.name: config[field.name] for field in fields(SwinEncoderConfig)})
        decoder   = PromptDecoderConfig(**{field.name: config[field.name] for field in fields(PromptDecoderConfig)})
        processor = ProcessorConfig(**{field.name: config[field.name] for field in fields(ProcessorConfig)})
        config    = PromptNougatModelConfig(
            encoder=encoder,
            decoder=decoder,
            processor=processor
        )
        return config, {}
        
@dataclass
class PromptNougatBaseConfig(PromptNougatModelConfig):
    model_config_name: Optional[str] = 'lorc_base'
    encoder: NormalSwinConfig
    decoder: BaseLOCRConfig
    processor: LougatProcessorConfig

@dataclass
class PromptNougatSmallConfig(PromptNougatModelConfig):
    model_config_name: Optional[str] = 'lorc_small'
    encoder: NormalSwinConfig
    decoder: SmallLOCRConfig
    processor: LougatProcessorConfig


@dataclass
class LlougatModelConfig(TrainerModelConfig):
    # below is only for simple parsing tracking
    encoder: SwinEncoderConfig
    decoder: LlougatDecoderConfig
    processor: ProcessorConfig
    
    def huggingface_dict(self):
        # assert self.model_config_name is not None, "Since the model_config_name is None, it seem you are not assign any model by --model [model] "
        # super().__init__(
        encoder = self.encoder.huggingface_dict()
        decoder = self.decoder.huggingface_dict() | {'image_size': self.encoder.image_size, 'image_embedding_size': self.encoder.image_embedding_size,
                                                     'vocab_size': len(self.processor.get_processor().tokenizer)}
        
        config = LlougatVEDConfig(encoder=encoder, decoder=decoder)

        return config

    @staticmethod
    def from_pretrained(path, **kargs):
        path = os.path.join(path, 'config.json')
        with open(path, 'r') as f:
            config = json.load(f)
        encoder   = SwinEncoderConfig( **{field.name: config[field.name] for field in fields(SwinEncoderConfig)})
        decoder   = LlougatDecoderConfig(**{field.name: config[field.name] for field in fields(PromptDecoderConfig)})
        processor = ProcessorConfig( **{field.name: config[field.name] for field in fields(ProcessorConfig)})
        config = LlougatModelConfig(
            encoder=encoder,
            decoder=decoder,
            processor=processor
        )
        return config, {}

    
@dataclass
class LlougatBaseConfig(LlougatModelConfig):
    model_config_name: Optional[str] = 'Llougat_base'
    encoder: NormalSwinConfig
    decoder: BaseLlougatConfig
    processor: LougatProcessorConfig

@dataclass
class LlougatSmallConfig(LlougatModelConfig):
    model_config_name: Optional[str] = 'Llougat_small'
    encoder: NormalSwinConfig
    decoder: SmallLlougatConfig
    processor: LougatProcessorConfig


@dataclass
class FlougatModelConfig(TrainerModelConfig):
    # below is only for simple parsing tracking
    encoder: FlorenceConfig
    decoder: LlougatDecoderConfig
    processor: FlougatProcessorConfig
    use_fixed_encoder: bool = False
    
    def huggingface_dict(self):
        # assert self.model_config_name is not None, "Since the model_config_name is None, it seem you are not assign any model by --model [model] "
        # super().__init__(
        encoder = self.encoder.huggingface_dict() | {'use_flashattn': self.decoder.attn_implementation == 'flash_attention_2'}

        decoder = self.decoder.huggingface_dict() | {'image_size': self.encoder.image_size, 'image_embedding_size': self.encoder.image_embedding_size,
                                                     'vocab_size': len(self.processor.get_processor().tokenizer)}

        config = LlougatVEDConfig(encoder=encoder, decoder=decoder)

        return config

    @staticmethod
    def from_pretrained(path, **kargs):
        path = os.path.join(path, 'config.json')
        with open(path, 'r') as f:
            config = json.load(f)
        encoder   = FlorenceConfig(**{field.name: config[field.name] for field in fields(SwinEncoderConfig)})
        decoder   = LlougatDecoderConfig(**{field.name: config[field.name] for field in fields(PromptDecoderConfig)})
        processor = ProcessorConfig(**{field.name: config[field.name] for field in fields(ProcessorConfig)})
        config = LlougatModelConfig(
            encoder=encoder,
            decoder=decoder,
            processor=processor
        )
        return config, {}

    def __post_init__(self):
        if self.encoder.projection_dim != self.decoder.hidden_size:
            print(f"WARNING:the dimension in encoder={self.encoder.projection_dim} is different from decoder={self.decoder.hidden_size}, we force the decoder dim to {self.encoder.projection_dim}")
            self.decoder.hidden_size = self.encoder.projection_dim
        

@dataclass
class FlougatBaseConfig(FlougatModelConfig):
    model_config_name: Optional[str] = 'Flougat_base'
    encoder: NormalFlorenceConfig
    decoder: BaseLlougatConfig
    processor: FlougatProcessorConfig


@dataclass
class FlougatSmallConfig(FlougatModelConfig):
    model_config_name: Optional[str] = 'Flougat_small'
    encoder: NormalFlorenceConfig
    decoder: SmallLlougatConfig
    processor: FlougatProcessorConfig

@dataclass
class FlougatXSmallConfig(FlougatSmallConfig):
    model_config_name: Optional[str] = 'FlougatX_small'
    encoder: FlorenceXConfig
    processor: FlougatXProcessorConfig

@dataclass
class UpArouGatSmallConfig(FlougatSmallConfig):
    model_config_name: Optional[str] = 'FlougatU_small'
    processor: UparxiveProcessorConfig


@dataclass
class UpArouGatBetaSConfig(FlougatSmallConfig):
    model_config_name: Optional[str] = 'FlougatU_betaS'
    encoder: Florence1KConfig
    decoder: SmallLlougatConfig
    processor: UparxiveBetaProcessorConfig

@dataclass
class UpArouGat1KSConfig(FlougatSmallConfig):
    model_config_name: Optional[str] = 'FlougatU1KS'
    encoder: Florence1KConfig
    decoder: SmallLlougatConfig
    processor: Uparxive1KProcessorConfig

@dataclass
class UpArouGat2KSConfig(FlougatSmallConfig):
    model_config_name: Optional[str] = 'FlougatU2KS'
    encoder: Florence2KConfig
    decoder: SmallLlougatConfig
    processor: UparxiveProcessor2kConfig

@dataclass
class UpArouGatBeta2KSConfig(FlougatSmallConfig):
    model_config_name: Optional[str] = 'FlougatU_B2KS'
    encoder: Florence2KConfig
    decoder: SmallLlougatConfig
    processor: UparxiveBetaProcessor2kConfig

@dataclass
class UpArouGatXSmallConfig(FlougatSmallConfig):
    model_config_name: Optional[str] = 'FlougatXU_small'
    encoder: FlorenceXConfig
    processor: UparxiveXProcessorConfig
    
@dataclass
class SlougatModelConfig(TrainerModelConfig):
    # below is only for simple parsing tracking
    encoder: SamEncoderConfigBase
    decoder: LlougatDecoderConfig
    processor: SlougatProcessorConfig
    use_fixed_encoder = False

    def huggingface_dict(self):
        # assert self.model_config_name is not None, "Since the model_config_name is None, it seem you are not assign any model by --model [model] "
        # super().__init__(
        encoder = self.encoder.huggingface_dict()
        decoder = self.decoder.huggingface_dict() | {'image_size': self.encoder.image_size, 
                                                     'image_embedding_size': self.encoder.image_embedding_size,
                                                     'vocab_size': len(self.processor.get_processor().tokenizer)}

        config = LlougatVEDConfig(encoder=encoder, decoder=decoder)

        return config

    @staticmethod
    def from_pretrained(path, **kargs):
        path = os.path.join(path, 'config.json')
        with open(path, 'r') as f:
            config = json.load(f)
        encoder   = FlorenceConfig(**{field.name: config[field.name] for field in fields(SwinEncoderConfig)})
        decoder   = LlougatDecoderConfig(**{field.name: config[field.name] for field in fields(PromptDecoderConfig)})
        processor = ProcessorConfig(**{field.name: config[field.name] for field in fields(ProcessorConfig)})
        config = LlougatModelConfig(
            encoder=encoder,
            decoder=decoder,
            processor=processor
        )
        return config, {}

    def __post_init__(self):
        if self.encoder.encoder_embedding_size != self.decoder.hidden_size:
            print(f"WARNING:the dimension in encoder={self.encoder.encoder_embedding_size} is different from decoder={self.decoder.hidden_size}, we force the decoder dim to {self.encoder.embed_dim}")
            self.decoder.hidden_size = self.encoder.encoder_embedding_size
        self.decoder.prompt_encoder_in_decoder = False
        

@dataclass
class SlougatBaseConfig(SlougatModelConfig):
    model_config_name: Optional[str] = 'Slougat_base'
    encoder: NormalSamEncoderConfig
    decoder: BaseLlougatConfig
    processor: Uparxive1KProcessorConfig


@dataclass
class SlougatSmallConfig(SlougatModelConfig):
    model_config_name: Optional[str] = 'Slougat_small'
    encoder: NormalSamEncoderConfig
    decoder: SmallLlougatConfig
    processor: Uparxive1KProcessorConfig


@dataclass
class RlougatModelConfig(TrainerModelConfig):
    # below is only for simple parsing tracking
    encoder: FlorenceConfig
    decoder: RlougatDecoderConfig
    processor: UparxiveProcessorConfig
    use_fixed_encoder: bool = False
    def huggingface_dict(self):
        # assert self.model_config_name is not None, "Since the model_config_name is None, it seem you are not assign any model by --model [model] "
        # super().__init__(
        encoder = self.encoder.huggingface_dict() | {'use_flashattn': self.decoder.attn_implementation == 'flash_attention_2'}

        decoder = self.decoder.huggingface_dict() | {'image_size': self.encoder.image_size, 'image_embedding_size': self.encoder.image_embedding_size,
                                                     'vocab_size': len(self.processor.get_processor().tokenizer)}

        config = RlougatVEDConfig(encoder=encoder, decoder=decoder)

        return config

    @staticmethod
    def from_pretrained(path, **kargs):
        path = os.path.join(path, 'config.json')
        with open(path, 'r') as f:
            config = json.load(f)
        encoder   = FlorenceConfig(**{field.name: config[field.name] for field in fields(SwinEncoderConfig)})
        decoder   = RlougatDecoderConfig(**{field.name: config[field.name] for field in fields(PromptDecoderConfig)})
        processor = ProcessorConfig(**{field.name: config[field.name] for field in fields(ProcessorConfig)})
        config = RlougatModelConfig(
            encoder=encoder,
            decoder=decoder,
            processor=processor
        )
        return config, {}

    def __post_init__(self):
        if self.encoder.projection_dim != self.decoder.hidden_size:
            print(f"WARNING:the dimension in encoder={self.encoder.projection_dim} is different from decoder={self.decoder.hidden_size}, we force the decoder dim to {self.encoder.projection_dim}")
            self.decoder.hidden_size = self.encoder.projection_dim
        

@dataclass
class RlougatBaseConfig(RlougatModelConfig):
    model_config_name: Optional[str] = 'Rlougat_base'
    encoder: NormalFlorenceConfig
    decoder: BaseRlougatConfig


@dataclass
class RlougatSmallConfig(RlougatModelConfig):
    model_config_name: Optional[str] = 'Rlougat_small'
    encoder: NormalFlorenceConfig
    decoder: SmallRlougatConfig

@dataclass
class Rlougat1KSConfig(RlougatModelConfig):
    model_config_name: Optional[str] = 'Rlougat_1KS'
    encoder: Florence1KConfig
    decoder: MidRlougatConfig
    processor: UparxiveBetaProcessorConfig


@dataclass
class FreezeConfig:
    freeze_encoder: Optional[str] = None
    freeze_decoder: Optional[str] = None

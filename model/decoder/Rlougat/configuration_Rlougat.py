
import warnings
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig, AutoModel
from typing import Optional, List, Union,Tuple
from transformers import VisionEncoderDecoderConfig
from transformers.models.llama.configuration_llama import LlamaConfig
### this is module for huggingface, those better construct it in hugging face style and wrapper it 


class RlougatConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`PromptBartConfig`]. It is used to instantiate a PromptBart
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the PromptBart
        [LLM4SCIENCE/Lougat/](https://huggingface.co/LLM4SCIENCE/Lougat) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "rlougat"

    def __init__(self,
        image_size:List[int]|Tuple[int, int] = None,
        image_embedding_size:List[int]|Tuple[int, int] = None,
        vocab_size=50000,#<<<---- default 5000
        max_position_embeddings=4096,#<<<---- default 4096
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        unk_token_id=3,
        is_encoder_decoder=True,
        decoder_start_token_id=0,#<<<---- default 0
        forced_eos_token_id=2,
        init_std=0.02,
        coordinate_retreive_method="position_revert",
        build_bbox_embedding_type="native",
        prompt_encoder_in_decoder = True,
        attn_implementation = "native",
        position_decoder_decay = 1,
        use_start_pos_embedding= False,
        pos_resolution = 2000,
        **kwargs,
    ):
        
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,  # <<<---- default 4096
            
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,

            **kwargs,
        )
        self.pos_resolution = pos_resolution
        self.position_decoder_decay = position_decoder_decay
        self.attn_implementation = attn_implementation
        self.build_bbox_embedding_type=  build_bbox_embedding_type
        self.prompt_encoder_in_decoder = prompt_encoder_in_decoder
        self.coordinate_retreive_method = coordinate_retreive_method
        self.init_std = init_std
        self.image_size = image_size
        self.image_embedding_size = image_embedding_size
        self.use_start_pos_embedding = use_start_pos_embedding

class RlougatVEDConfig(VisionEncoderDecoderConfig):
    model_type = "rlougat_ved"
    

AutoConfig.register("rlougat", RlougatConfig)
AutoConfig.register("rlougat_ved", RlougatVEDConfig)

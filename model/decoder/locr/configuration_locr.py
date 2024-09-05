
import warnings
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig, AutoModel
from typing import Optional, List, Union,Tuple
from transformers import VisionEncoderDecoderConfig

### this is module for huggingface, those better construct it in hugging face style and wrapper it 
class PromptBartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PromptBartConfig`]. It is used to instantiate a PromptBart
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the PromptBart
        [LLM4SCIENCE/Lougat/](https://huggingface.co/LLM4SCIENCE/Lougat) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "promptbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    pruned_heads = {}
    
    def __init__(
        self,
        vocab_size=50000,#<<<---- default 5000
        max_position_embeddings=4096,#<<<---- default 4096
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        init_std=0.02,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        classifier_dropout=0.0,
        scale_embedding=True, #<<<---- default True
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        unk_token_id=3,
        is_encoder_decoder=True,
        decoder_start_token_id=0,#<<<---- default 0
        forced_eos_token_id=2,
        # ----------- above is almost same as transformers.BartConfig  -------------
        # ----------- below is new for PromptBartConfig  -------------
        position_decoder_hidden_size:int = 256,
        position_decoder_layers:int = 3,
        position_decoder_decay:float = 1,
        add_cross_attention:bool = True,
        prompt_embed_dim:int=1024, #,256,
        omit_ratio: float = 0,
        BartAttention_implement: str = 'sdpa',
        is_decoder= True,
        image_size:List[int]|Tuple[int, int] = None,
        image_embedding_size:List[int]|Tuple[int, int] = None,
        use_image_bias: bool = False,
        **kwargs,
    ):
        # assert image_size is not None, "please assign image_size in config"
        # assert image_embedding_size is not None, "please assign image_embedding_size in config"
        self.use_image_bias= use_image_bias
        self.image_size = image_size
        self.image_embedding_size = image_embedding_size
        # -----------------------------
        self.is_decoder= is_decoder ## <--default True
        self.is_encoder_decoder= is_encoder_decoder # <--default True to get cross-attention
        # -----------------------------
        self.position_decoder_hidden_size = position_decoder_hidden_size
        self.position_decoder_layers = position_decoder_layers
        self.position_decoder_decay = position_decoder_decay
        self.add_cross_attention = add_cross_attention
        self.prompt_embed_dim = prompt_embed_dim
        self.omit_ratio = omit_ratio
        self.BartAttention_implement = BartAttention_implement
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
       
        # ------------------ dropout ---------------------
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        # --------------------------------------------------

        self.activation_function = activation_function
        self.init_std = init_std
        
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.unk_token_id = unk_token_id
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            is_encoder_decoder=is_encoder_decoder,
            is_decoder= is_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        # ensure backward compatibility for BART CNN models
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )

class LougatConfig(VisionEncoderDecoderConfig):
    model_type = "lougat"
    

AutoConfig.register("promptbart", PromptBartConfig)
AutoConfig.register("lougat", LougatConfig)
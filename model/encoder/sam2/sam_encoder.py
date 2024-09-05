import torch
from typing import Optional,List ,Dict
from transformers import AutoModelForCausalLM, AutoModel,AutoConfig,PreTrainedModel
from .modeling_sam2 import Sam2ImageEncoder, Sam2ImageEncoderConfig, Sam2LayerNorm2d
from einops import rearrange
import os
from transformers.utils import logging
logger = logging.get_logger(__name__)

class SamFeatureConfig(Sam2ImageEncoderConfig):
    model_type = "sam2_feature"
    compile_image_encoder= True
    pretrain_weight_path = "pretrain_weights/sam2_image_encoder.pt"
    def __init__(self, encoder_embedding_size,**kwargs):
        super().__init__(**kwargs)
        self.encoder_embedding_size = encoder_embedding_size
    
class SamFeatureEncoder(PreTrainedModel):
    config_class = SamFeatureConfig
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    def __init__(self, config:SamFeatureConfig):
        super().__init__(config)
        self.sam_image_encoder = Sam2ImageEncoder(config)
        if not os.path.exists(config.pretrain_weight_path):
            logger.warning_once("The pretrain weight path is not exist. Please know it is not suggested to train Sam model from sketch. Make suer you load it")
        else:
            self.sam_image_encoder.load_state_dict(torch.load(config.pretrain_weight_path))
        ## input is (1, 256, 64, 48) 
        ## up feature to (1, encoder_embedding_size, 32, 24) which use a cnn layer with kernel size 3, stride 2 and padding 1
        self.feature_up = torch.nn.Sequential(
            torch.nn.Conv2d(config.d_model, config.encoder_embedding_size, kernel_size=3, stride=2, padding=1),
            Sam2LayerNorm2d(config.encoder_embedding_size),
            torch.nn.ReLU(),
        )
    def forward(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            sam2_feature = self.sam_image_encoder(pixel_values)
        sam2_feature = sam2_feature['vision_features']
        sam2_feature = self.feature_up(sam2_feature)
        sam2_feature = rearrange(sam2_feature, 'b c h w -> b (h w) c')
        return sam2_feature

AutoConfig.register("sam2_feature", SamFeatureConfig)
AutoModel.register(SamFeatureConfig, SamFeatureEncoder)

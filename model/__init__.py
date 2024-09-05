from .model_arguements import TrainerModelConfig
from .trainer_model import PromptNougatModel, LlougatTrainerModel,RlougatTrainerModel
from .replace_flash_attn import replace_promptdecoder_attn_with_flash_attn

replace_promptdecoder_attn_with_flash_attn('OnlySwin')


def load_model(config: TrainerModelConfig):
    """
    Build the model script only when using
    """
    if 'rlougat' in config.model_config_name.lower():
        model =  RlougatTrainerModel(config)
    elif 'lougat' in config.model_config_name.lower():
        model =  LlougatTrainerModel(config)
    else:
        model =  PromptNougatModel(config)

    return model 





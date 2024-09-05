import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import re
from model.model_arguements import *
from model.decoder.locr.configuration_locr import SmallPromptBartConfig
from model import load_model
from simple_parsing import ArgumentParser

parser = ArgumentParser()
parser.add_arguments(PromptNougatSmallConfig, dest="config")
args = parser.parse_args(args=[])
config = args.config

(image_tensors, input_ids,attention_masks,label_ids,prompts) = torch.load('debug/inputs.pt')
config.decoder.BartAttention_implement = 'sdpa'
model = load_model(config)
import torch
from utils import smart_load_weight
weight=torch.load('/mnt/data/oss_beijing/sunyu/nougat/PromptNougat/result/nougat/20240207/pytorch_model.alpha.bin',
                  map_location='cpu')
smart_load_weight(model,weight,strict=False, shape_strict=True)
_ = model.cuda()
_ = model.eval()
from model.trainer_model import *
pretrained_model = model
model.config.max_length = 4096
ground_truth = pretrained_model.decoder.processor.batch_decode(label_ids, skip_special_tokens=True)
with torch.cuda.amp.autocast(dtype=torch.float32):
    outputs = pretrained_model.inference(
            image_tensors=image_tensors,    
            input_ids = input_ids,
            attention_mask=attention_masks,
            return_attentions=True,
            prompt=prompts[:,:-1,:,:].clone(),
            validation=True,
        )
print(outputs['predictions'][0])
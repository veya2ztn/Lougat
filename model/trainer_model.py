

import os
import torch.nn as nn
from typing import List, Optional
import numpy as np
import torch
from PIL import Image
import logging
from einops import rearrange
from .encoder.florence.modeling_florence2 import Florence2VisionModelWithProjection
from .decoder.locr.modeling_locr import LougatForVision2Seq
from .decoder.llougat.modeling_llougat import LlougatForVision2Seq, Union, Cache
from .decoder.Rlougat.modeling_Rlougat import RlougatForVision2Seq
from .model_arguements import PromptNougatModelConfig, LlougatModelConfig, RlougatModelConfig
from .model_arguements import FreezeConfig
from .cal_loss import cal_loss, cal_loss_fast,celoss,diou_loss,CE
from transformers import AutoModel,AutoModelForCausalLM
from dataset.resource import type_mapping
def compute_full_loss(error_pool):
    weights = {'loss_start':1, 'loss_math':1, 'loss_table':1, 'loss_txt':1} ## <-- better assign weight outside the loop
    weight_token,weight_position = 5, 1
    weight_diou, weight_cls = 0.3,1


    loss_token = 0
    weight_sum = 0
    for key, weight in weights.items():
        if error_pool[key] is None:continue
        loss_token  = loss_token + error_pool[key]
        weight_sum += weight
    loss_token  = loss_token/weight_sum
    

    iou = error_pool.get('iou',None)
    diou= error_pool.get('diou',None)
    fl = error_pool.get('fl', None)
    if iou is not None:
        diou = 0 if diou is None else diou
        fl = 0 if fl is None else fl
        loss_position = weight_diou*diou + weight_cls*fl
        loss = (weight_token*loss_token + weight_position*loss_position)/(weight_token+weight_position)
        error_pool = error_pool|{'loss_position':loss_position}
    else:
        loss = loss_token
    return loss
dtype_pool={
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
class BaseTrinerModel:

    def get_loss(self, logits,labels,preded_bbox,bboxes_true, heat_map, attention_map, type_mask):
        loss_pool = cal_loss_fast(logits[type_mask][attention_map[type_mask]],
                                  labels[type_mask][attention_map[type_mask]],
                                  preded_bbox[type_mask][attention_map[type_mask]],
                                  bboxes_true[type_mask][attention_map[type_mask]], 
                                  heat_map[type_mask][attention_map[type_mask]] if heat_map is not None else None)
        loss_now  = self.token_weight*loss_pool['token_loss'] + self.bbox_weight*loss_pool['bbox_diou'] + self.bbox_weight*loss_pool.get('hm_loss',0)
        loss_now  = loss_now/(self.token_weight + self.bbox_weight + 1e-5)
        return loss_now, loss_pool

    def freeze_model_during_train(self, freeze_config:Optional[FreezeConfig]=None):
        if freeze_config is None: return 
        if freeze_config.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_config.freeze_decoder:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(  # <--- use this genereate better result then use default
        self,
        image_tensors: Optional[torch.Tensor] = None,
        pre_input_ids: Optional[torch.LongTensor] = None,
        prompt_in: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        label_id: Optional[torch.LongTensor] = None,
        prompt_true: Optional[torch.Tensor] = None,
        label_token_types: Optional[torch.LongTensor] = None,
    ):
        
        input_ids = pre_input_ids
        labels = label_id
        input_bboxes=prompt_in
        bboxes_true = prompt_true
        image_embedding = None
        if self.use_fixed_encoder:
            with torch.inference_mode():
                self.encoder.eval()
                image_embedding = self.encoder(image_tensors)
        decoder_outputs = super().forward(
            image_tensors = image_tensors,
            input_ids = input_ids,
            input_bboxes = input_bboxes,
            attention_mask = attention_mask,
            return_dict=False,
            image_embedding=image_embedding,
        )
        if labels is None:
            return None, {}, decoder_outputs
        
        logits, preded_bbox = decoder_outputs[:2]
        loss_txt, loss_math, loss_table, loss_start, diou, iou, fl = None, None, None, None, None, None, None
        attention_mask = attention_mask.bool()

        # loss_txt, loss_math, loss_table, loss_start, diou, iou, fl = cal_loss(
        #     bs=logits.shape[0],
        #     logits=logits.reshape(-1, self.decoder.config.vocab_size), 
        #     labels=labels.reshape(-1),
        #     prompt_pred=preded_bbox,
        #     prompt_true=bboxes_true)
        
        # return (loss_txt, loss_math, loss_table, loss_start, diou, iou, fl), decoder_outputs
        hm = None
        if isinstance(preded_bbox, tuple):
            (preded_bbox,hm) = preded_bbox

        final_loss_pool = {}



        start_idx = self.start_count
        start_map = torch.zeros_like(attention_mask)
        start_map[:,:start_idx] = True
        loss_start,loss_start_pool= self.get_loss(logits,labels,preded_bbox,bboxes_true, hm, attention_mask, start_map)
        loss_start_pool = {"e_"+"start_"+key:val for key,val in loss_start_pool.items()}|{f"start_loss":loss_start}
        final_loss_pool = final_loss_pool | loss_start_pool
        loss_full = 0
        for type_id, type_name in self.ids_to_type_mapping.items():
            type_map = (label_token_types == type_id)
            if not type_map.any(): 
                loss_pool = {f"{type_name}_loss":torch.FloatTensor([0])}
                final_loss_pool = final_loss_pool | loss_pool
                continue
            loss_now, loss_pool = self.get_loss(logits,labels,preded_bbox,bboxes_true, hm, attention_mask, type_map)
            loss_pool = {"e_"+type_name+"_"+key:val for key,val in loss_pool.items()}|{f"{type_name}_loss":loss_now}
            final_loss_pool = final_loss_pool | loss_pool
            loss_full+=loss_now
    
        loss = (self.start_weight*loss_start + self.full_weight*loss_full)/(self.start_weight+self.full_weight+1e-5)

        
        return loss, final_loss_pool, decoder_outputs

    def auto_init(self, config):
        huggingface_config = config.huggingface_dict()
        huggingface_config._attn_implementation = config.decoder.attn_implementation
        encoder_attn_implementation = 'eager' if 'swin' in huggingface_config.encoder.model_type else config.decoder.attn_implementation
        encoder = AutoModel.from_config(huggingface_config.encoder, attn_implementation=encoder_attn_implementation)
        super().__init__(config= huggingface_config, encoder= encoder)
        self.processor = config.processor.get_processor()
        self.bbox_weight = config.loss_config.bbox_weight
        self.token_weight= config.loss_config.token_weight
        self.start_count = config.loss_config.start_count
        self.start_weight= config.loss_config.start_weight
        self.full_weight = config.loss_config.full_weight
        self.ids_to_type_mapping = {0:"text", 1:"math", 2:"floats"}
        

class PromptNougatModel(LougatForVision2Seq):
    def __init__(self, config:PromptNougatModelConfig):
        super().__init__(config.huggingface_dict())
        self.processor = config.processor.get_processor()
        self.bbox_weight = config.loss_config.bbox_weight
        self.token_weight= config.loss_config.token_weight
        self.start_count = config.loss_config.start_count
        self.start_weight= config.loss_config.start_weight
        self.full_weight = config.loss_config.full_weight
    def forward(  # <--- use this genereate better result then use default
        self,
        image_tensors: Optional[torch.Tensor] = None,
        pre_input_ids: Optional[torch.Tensor] = None,
        prompt_in: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        label_id: Optional[torch.Tensor] = None,
        prompt_true: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        full_prompt_in: Optional[torch.Tensor] = None,
        **kargs,
    ):
        input_ids = pre_input_ids
        labels = label_id
        input_bboxes=prompt_in
        bboxes_true = prompt_true
        # if labels is None     : labels = input_ids[:,1:]
        # if bboxes_true is None: bboxes_true = input_bboxes[:,1:]

        decoder_outputs = super().forward(
            image_tensors=image_tensors,
            pre_input_ids=pre_input_ids,
            attention_mask=attention_mask,
            label_id=label_id,
            prompt_in=prompt_in,
            prompt_true=prompt_true,
            current_step=None,
            full_prompt_in=full_prompt_in,
            **kargs
        )
        if labels is None:
            return None, {}, decoder_outputs
        logits, preded_bbox = decoder_outputs[:2]
        attention_mask = attention_mask.bool()
        
        
        # loss_txt, loss_math, loss_table, loss_start, diou, iou, fl = cal_loss(
        #     bs=logits.shape[0],
        #     logits=logits.reshape(-1, self.decoder.config.vocab_size), 
        #     labels=labels.reshape(-1),
        #     prompt_pred=preded_bbox,
        #     prompt_true=bboxes_true)
        # error_pool = {
        #     'loss_txt':loss_txt, 'loss_math':loss_math, 'loss_table':loss_table, 'loss_start':loss_start,
        #     'diou':diou, 'iou':iou, 'fl':fl
        # }
        # loss = compute_full_loss(error_pool)
        # return loss, error_pool, decoder_outputs
        
        (preded_bbox,hm) = preded_bbox

        start_idx         = self.start_count
        logits_start      = logits[:,:start_idx][attention_mask[:,:start_idx]]
        labels_start      = labels[:,:start_idx][attention_mask[:,:start_idx]]
        preded_bbox_start = preded_bbox[:,:start_idx][attention_mask[:,:start_idx]]
        bboxes_true_start = bboxes_true[:,:start_idx][attention_mask[:,:start_idx]]
        heat_map_start    = hm[:,:start_idx][attention_mask[:,:start_idx]]
        loss_start_pool   = cal_loss_fast(logits_start,labels_start,preded_bbox_start,bboxes_true_start, heat_map_start)

        loss_start = self.token_weight*loss_start_pool['token_loss'] + self.bbox_weight*loss_start_pool['bbox_diou'] + self.bbox_weight*loss_start_pool['hm_loss']
        loss_start = loss_start/(self.token_weight+self.bbox_weight+1e-5)


        logits           = logits[attention_mask]
        labels           = labels[attention_mask]
        preded_bbox      = preded_bbox[attention_mask]
        bboxes_true      = bboxes_true[attention_mask]
        heat_map         = hm[attention_mask]
        loss_full_pool   = cal_loss_fast(logits,labels,preded_bbox,bboxes_true, heat_map)

        loss_full = self.token_weight*loss_full_pool['token_loss'] + self.bbox_weight*loss_full_pool['bbox_diou'] + self.bbox_weight*loss_full_pool['hm_loss']
        loss_full = loss_full/(self.token_weight+self.bbox_weight+1e-5)

        loss = (self.start_weight*loss_start + self.full_weight*loss_full)/(self.start_weight+self.full_weight+1e-5)

        loss_start_pool = {"start_"+key:val for key,val in loss_start_pool.items()}
        return loss, loss_start_pool|loss_full_pool, decoder_outputs
    
    def inference(
        self,
        image: Image.Image = None,
        image_tensors: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = True,
        pdf = None,
        prompt = None,
        validation = False,
        use_cache = True,
        current_step = None,
        max_new_tokens = None,
    ):
        """
        Generate a token sequence in an auto-regressive manner.

        Args:
            image: input document image (PIL.Image)
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
        """
        output = {
            "predictions": list(),
            "sequences": list(),
            "repeats": list(),
            "repetitions": list(),
            'logits': list(),
            "prompt_pred":list(),
        }
        if image is None and image_tensors is None:
            logging.warn("Image not found")
            return output
        if image_tensors is None:
            image_tensors = self.processor([image], return_tensors="pt").pixel_values
        output = self.generate(image_tensors, only_return_sequences=False,max_new_tokens=max_new_tokens)
        output["repetitions"] = self.processor.batch_decode(output["repetitions"], skip_special_tokens=True)
        # for b in range(len(output["repetitions"])):
        #     if output["repeats"][b]:
        #         print(f'Rep {b}_{output["repeats"][b]}: {output["repetitions"][b]}')
        # validation=True时,output['sequence']==output['predictions']==gt不能作为评价指标,必须decode(logits)
        #output["predictions"] = postprocess(self.processor.batch_decode(output["sequences"], skip_special_tokens=True),markdown_fix=True,)
        output["predictions"] = self.processor.post_process_generation(self.processor.batch_decode(output["sequences"], skip_special_tokens=True), fix_markdown=True)
        return output

    def freeze_model_during_train(self, freeze_config:Optional[FreezeConfig]=None):
        if freeze_config is None: return 
        if freeze_config.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_config.freeze_decoder:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False

class LlougatTrainerModel(BaseTrinerModel,LlougatForVision2Seq):
    def __init__(self, config: LlougatModelConfig):
        huggingface_config = config.huggingface_dict()
        huggingface_config._attn_implementation = config.decoder.attn_implementation
        encoder_attn_implementation = 'eager' if 'swin' in huggingface_config.encoder.model_type else config.decoder.attn_implementation
        encoder_attn_implementation = 'eager' if config.use_fixed_encoder else encoder_attn_implementation
        self.use_fixed_encoder = config.use_fixed_encoder
        #torch_dtype = torch.bfloat16 if encoder_attn_implementation == "flash_attention_2" else torch.float32 
        encoder = AutoModel.from_config(huggingface_config.encoder, attn_implementation=encoder_attn_implementation) 
        #decoder = AutoModelForCausalLM.from_config(huggingface_config.decoder, attn_implementation=encoder_attn_implementation,torch_dtype=torch_dtype)
        #### <----- do not use torch_dtype when training, cause train OOM (I dont know why)
        super().__init__(config= huggingface_config, encoder= encoder)
        self.processor = config.processor.get_processor()
        self.bbox_weight = config.loss_config.bbox_weight
        self.token_weight= config.loss_config.token_weight
        self.start_count = config.loss_config.start_count
        self.start_weight= config.loss_config.start_weight
        self.full_weight = config.loss_config.full_weight
        self.ids_to_type_mapping = {0:"text", 1:"math", 2:"floats"}

    
class RlougatTrainerModel(BaseTrinerModel,RlougatForVision2Seq):
    def __init__(self, config: RlougatModelConfig):
        huggingface_config = config.huggingface_dict()
        huggingface_config._attn_implementation = config.decoder.attn_implementation
        encoder_attn_implementation = 'eager' if 'swin' in huggingface_config.encoder.model_type else config.decoder.attn_implementation
        encoder_attn_implementation = 'eager' if config.use_fixed_encoder else encoder_attn_implementation
        self.use_fixed_encoder = config.use_fixed_encoder
        encoder = AutoModel.from_config(huggingface_config.encoder, attn_implementation=encoder_attn_implementation)
        super().__init__(config= huggingface_config, encoder= encoder)
        self.processor = config.processor.get_processor()
        self.bbox_weight = config.loss_config.bbox_weight
        self.token_weight= config.loss_config.token_weight
        self.start_count = config.loss_config.start_count
        self.start_weight= config.loss_config.start_weight
        self.full_weight = config.loss_config.full_weight
        self.ids_to_type_mapping = {0:"text", 1:"math", 2:"floats"}

    @staticmethod
    def cal_loss_fast(logits,labels,prompt_pred,prompt_true,heat_map_x, heat_map_y):
        BL1, vocab_size = logits.shape
        BL2, =  labels.shape
        
        BL3, D11,D12 = prompt_pred.shape
        BL4, D21,D22 = prompt_true.shape
        assert BL1 == BL2 and BL1 == BL3 and BL1 == BL4
        assert D11 == D12 and D11 == D21 and D11 == D22 and D11 == 2
        diou,iou = diou_loss(pred=prompt_pred,target=prompt_true,pre5pos=None)  
        
        loss_pool =  {
            "token_loss": CE(logits, labels),
            "bbox_diou" : diou, 
            "bbox_iou"  : iou,
        }
        
        c_x_true= prompt_true[...,0].mean(-1) # (B, L )
        BL, D = heat_map_x.shape
        BL    = c_x_true
        #print(c_x_true.max())
        c_x_true=torch.clamp((c_x_true*(D-1)).long(),0,D-1) # 0->1999 ## <--- may error for bfloat16
        #print(c_x_true.max())
        loss_pool["hm_loss_x"]  = CE(heat_map_x, c_x_true)

        c_y_true= prompt_true[...,1].mean(-1) # (B, L )
        BL, D = heat_map_y.shape
        BL    = c_y_true
        #print(c_y_true.max())
        c_y_true=torch.clamp((c_y_true*(D-1)).long(),0,D-1) # 0->1999
        #print(c_y_true.max())
        loss_pool["hm_loss_y"]  = CE(heat_map_y, c_y_true)

        loss_pool["hm_loss"] = (loss_pool["hm_loss_x"] + loss_pool["hm_loss_y"])
        return loss_pool
    
    
    def get_loss(self, logits,labels,preded_bbox,bboxes_true, heat_map, attention_map, type_mask):
        heat_map_cx = heat_map['c_x_head_logits'] #(B, L , 2000)
        heat_map_cy = heat_map['c_y_head_logits'] #(B, L , 2000)
        loss_pool = self.cal_loss_fast(logits[type_mask][attention_map[type_mask]],
                                  labels[type_mask][attention_map[type_mask]],
                                  preded_bbox[type_mask][attention_map[type_mask]],
                                  bboxes_true[type_mask][attention_map[type_mask]], 
                                  heat_map_cx[type_mask][attention_map[type_mask]],
                                  heat_map_cy[type_mask][attention_map[type_mask]])
        loss_now  = self.token_weight*loss_pool['token_loss'] + self.bbox_weight*loss_pool['bbox_diou'] + self.bbox_weight*loss_pool.get('hm_loss',0)
        loss_now  = loss_now/(self.token_weight + self.bbox_weight + 1e-5)
        return loss_now, loss_pool
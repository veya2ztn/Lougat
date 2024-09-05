"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
from .dataset_arguements import NougatDatasetConfig, LocrDatasetConfig, UparxiveDatasetConfig, ParquetUpaxiveConfig
from .utils import Timers
import os
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
from math import prod
import random
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

import numpy as np
from PIL import Image,ImageOps
import cv2
from torchvision.transforms.functional import resize, rotate
#from nougat.transforms import train_transform, test_transform
from .transforms import train_transform, test_transform
from .resource import SciPDFDataset, LougatDataset
from tqdm.auto import tqdm
import base64
from io import BytesIO
from transformers.models.nougat import NougatProcessor, NougatImageProcessor
from transformers import AutoProcessor
from .resource_utils import fill_empty_bbox_follow_last
from typing import List
class NougatDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """

    def __init__(self, dataset_path, split:str, processor:AutoProcessor, config:NougatDatasetConfig, enable_Timer = False, dummy=False):
        super().__init__()
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length          = config.max_length
        self.prompt_label_length = config.prompt_label_length
        self.split = split
        self.perturb = "NOUGAT_PERTURB" in os.environ and os.environ["NOUGAT_PERTURB"]
        # TODO improve naming conventions
        template = "%s"
        assert not processor.image_processor.do_pad or "bboxes" in processor.image_processor.model_input_names, "currently, please do not change pad/crop image via the bbox is located by the a*width/height"
        assert not processor.image_processor.do_crop_margin or "bboxes" in processor.image_processor.model_input_names, "currently, please do not change pad/crop image via the bbox is located by the a*width/height"

        if processor.image_processor.rescale_factor*255 != 1: 
            print("WARNING: the image rescale from your processor will not lead the pixel value 0->0 and 255->1, be careful")
        if processor.image_processor.image_mean != [0.485,0.456,0.406]:
            print(f"WARNING: the image normlize from your processor is not a standard normlize_mean=[0.485,0.456,0.406]<= yours = {processor.image_processor.image_mean}")
        if processor.image_processor.image_std != [0.229,0.224,0.225]:
            print(f"WARNING: the image normlize from your processor is not a standard normlize_std=[0.229,0.224,0.225] <= yours = {processor.image_processor.image_std}")
        if isinstance(config, UparxiveDatasetConfig):
            self.dataset =LougatDataset(dataset_path, split=self.split, template=template, root_name=config.root_name,random_dpi=config.random_dpi,tokenizer=self.tokenizer) 
            self.dataset_length = len(self.dataset)
        elif isinstance(config, ParquetUpaxiveConfig):
            from datasets import load_dataset
            if isinstance(dataset_path,str) and os.path.isdir(dataset_path):
                dataset_path = [os.path.join(dataset_path,t) for t in os.listdir(dataset_path)]
            # split_alias = {'train':'train','validation':'valid','test':'test'}
            # split=split_alias[split]
            self.dataset = load_dataset("parquet",data_files = dataset_path )['train']
            self.dataset_length = len(self.dataset)
        else:
            self.dataset = SciPDFDataset(dataset_path, split=self.split, template=template, root_name=config.root_name) 
            self.dataset_length = len(self.dataset)

        self.pad_id          = self.tokenizer.pad_token_id
        self.eos_id          = self.tokenizer.eos_token_id
        self.global_start_id = config.decoder_start_token_id   # <work>:在PromptBartConfig初始化处改
        self.gloabl_eos_id   = 10     # </work>
        if self.perturb:
            print("Perturb")

        self.training = split.lower() =='train'
        self.timers = Timers(enable_Timer)
        self.dummy  = dummy
        self.example_data = None
        self.force_use_augmentation = False
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def __len__(self) -> int:
        return self.dataset_length

    @property
    def to_tensor(self):
        if self.training:
            return train_transform
        else:
            return test_transform

    def crop_margin(self, img: Image.Image) -> Image.Image:
        with self.timers('prepare_input/crop_margin/data'):
            data = np.array(img.convert("L"))
            data = data.astype(np.uint8)
        with self.timers('prepare_input/crop_margin/computer1'):
            max_val = data.max()
            min_val = data.min()
        if max_val == min_val:
            return img
        with self.timers('prepare_input/crop_margin/computer2'):
            data = (data - min_val) / (max_val - min_val) * 255
            #data = np.interp(data, (min_val, max_val), (0, 255))
        with self.timers('prepare_input/crop_margin/binary'):
            gray = 255 * (data < 200).astype(np.uint8)
        with self.timers('prepare_input/crop_margin/findNonZero'):
            coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        with self.timers('prepare_input/crop_margin/boundingRect'):
            a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        with self.timers('prepare_input/crop_margin/crop'):
            out = img.crop((a, b, w + a, h + b))
        return out

    def _pre_processing(self, img: Image.Image, random_padding: bool = False) -> Image:
        with self.timers('prepare_input/convert'):
            img = img.convert("RGB")

        with self.timers('prepare_input/crop_margin'):
            try:
                img = self.crop_margin(img)
            except OSError:
                # might throw an error for broken files
                return
        if img.height == 0 or img.width == 0:
            return

        with self.timers('prepare_input/rotate'):
            if self.config.align_long_axis and (
                (self.config.input_size[0] >
                 self.config.input_size[1] and img.width > img.height)
                or (self.config.input_size[0] < self.config.input_size[1] and img.width < img.height)
            ):
                img = rotate(img, angle=-90, expand=True)

        with self.timers('prepare_input/resize'):
            img = resize(img, min(self.config.input_size))

        with self.timers('prepare_input/thumbnail'):
            img.thumbnail(
                (self.config.input_size[1], self.config.input_size[0]))
            delta_width = self.config.input_size[1] - img.width
            delta_height = self.config.input_size[0] - img.height

        with self.timers('prepare_input/padding'):
            if random_padding:
                pad_width = np.random.randint(low=0, high=delta_width + 1)
                pad_height = np.random.randint(low=0, high=delta_height + 1)
            else:
                pad_width = delta_width // 2
                pad_height = delta_height // 2
            padding = (
                pad_width,
                pad_height,
                delta_width - pad_width,
                delta_height - pad_height,
            )

        with self.timers('prepare_input/expand'):
            img = ImageOps.expand(img, padding)

        return img
    
    def prepare_input(
        self, img: Image.Image, bboxes: np.ndarray, random_padding: bool = False
    ) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        ### use process
        if img is None:
            return

        with self.timers('prepare_input'):
            #img = self._pre_processing(img, random_padding)
            if hasattr(self.processor.image_processor, "augmentation_image_processing"):
                width, height= img.size
                bboxes = bboxes.reshape(-1,4)
                bboxes = np.clip(bboxes, 0, 1)
                bboxes[:,[0,2]]*=width
                bboxes[:,[1,3]]*=height
                if self.training or self.force_use_augmentation:
                    output = self.processor.image_processor.augmentation_image_processing([img.convert('RGB')], [bboxes],
                                                                                            random_gray_method=False,
                                                                                            random_resize_std=0.1,
                                                                                            random_pad_pixels=200,
                                                                                            random_crop_margin=False,
                                                                                        do_rescale=False,
                                                                                        do_normalize=False,
                                                                                        )
                else:
                    output = self.processor.image_processor.preprocess([img.convert('RGB')], [bboxes],do_rescale=False,do_normalize=False,)
                img    = output.pixel_values[0].transpose(1, 2, 0)
                bboxes = output.bboxes[0]
                height,width,  _ = img.shape  #<---- careful this, make sure which line is height and which one is width
                #width,height, _ = img.shape 
                bboxes[:,[0,2]]/=width
                bboxes[:,[1,3]]/=height
                bboxes = bboxes.reshape(-1,2,2)
                # for bbox in bboxes:
                #     if bbox.max()>1:
                #         torch.save({'image':img, 'bboxes':bboxes},"debug.pt")
                #         raise
                
            else:
                img = self.processor(img.convert('RGB'), do_rescale=False,do_normalize=False).pixel_values[0].transpose(1, 2, 0)
        with self.timers('augmentation'):
            out = self.to_tensor(img)
        return out,bboxes
    

    def format_data(self,sample):
        if isinstance(sample.get('image',None),str):
            sample["image"] = Image.open(BytesIO(base64.b64decode(sample['image'])))
       
        
     
        if "token" not in sample and 'token_ids' in sample:
            sample['token'] = sample['token_ids']
        if "token_type" not in sample and ("text_type" in sample or "text_types" in sample):
            text_type = sample.get("text_type",None) or sample.get("text_types",None)
            sample['token_type'] = text_type
        if "prompt"  not in sample and ("bbox" in sample or "bboxes" in sample):
            bboxes = sample.get("bbox",None) or sample.get("bboxes",None)
            bboxes = fill_empty_bbox_follow_last(bboxes)
            bboxes = np.array(bboxes)
            bboxes = np.clip(bboxes,0,1)
            sample['prompt']=bboxes
        if sample is None or sample["image"] is None or prod(sample["image"].size) == 0:
            input_tensor = None
        else:
            input_tensor,sample['prompt'] = self.prepare_input(sample["image"], sample['prompt'], random_padding= self.split == "train")

        with self.timers('tokenize'):
            if not sample.get("token", None):
                tokenizer_out = self.tokenizer(sample["pretext"],return_token_type_ids=False,return_length=True)

                # length = tokenizer_out['length']
                # if max(length) >= 4096:
                #     print(sample['meta'])
                #     raise 

                # if len(length) >=4096:
                #     print(sample['meta'])
                #     raise 
            else:
                token = sample["token"]
                tokenizer_out = {"input_ids":token}
    
        # 全文prompt，len(pretext)=len(prompt)，len(label)=0 -> len(label)=len(input_ids)    
        
        
        
        with self.timers('polish'):

            pre_ids    = [self.tokenizer.bos_token_id, self.tokenizer.bos_token_id] 
            token_type = [0,0]
            prompts    = [[[0,0],[0,0]],
                            ]
            
            pre_id_lst = tokenizer_out["input_ids"]    # lst:[[0,toekn1_1,token1_2,2],[0,token2_1,2],[0,token3_1,token3_2,2]]
            for i,pre_id in enumerate(pre_id_lst):  
                if pre_id[0] == self.tokenizer.bos_token_id and pre_id[-1] == self.tokenizer.eos_token_id:
                    pre_id = pre_id[2:-1] if pre_id[1] == 243  else pre_id[1:-1]   # 去掉<s>和</s>
                pre_ids.extend(pre_id)    
                prompts.extend([sample['prompt'][i]]*len(pre_id))# 一个word可能被拆分成多个token
                token_type_id = sample['token_type'][i] if 'token_type' in sample else 0
                token_type.extend([token_type_id]*len(pre_id))
            # truncation
            pre_ids        = pre_ids[:self.max_length-1]    + [self.tokenizer.eos_token_id]
            token_type     = token_type[:self.max_length-1] + [-1]
            prompts        = prompts[:self.max_length-2] + [[[0.99,0.99],[1,1]], [[0.99,0.99],[1,1]]]
            
            attention_mask = [1]*(len(pre_ids) - 1) ### a casual mask should be from the 0 th item to the n - 1 item
            
            # padding
            pre_ids        = pre_ids        + [self.pad_id]  *max(0,self.max_length-len(pre_ids))
            token_type     = token_type     + [-1           ]*max(0,self.max_length-len(token_type))
            prompts        = prompts        + [[[0,0],[0,0]]]*max(0,self.max_length-len(prompts))
            
            attention_mask = attention_mask + [0]*max(0,self.max_length-len(attention_mask))
            
            # to_tensor
            
            token_ids      = torch.LongTensor(pre_ids)
            token_types    = torch.LongTensor(token_type)
            prompts        = torch.FloatTensor(np.array(prompts))         # 这里的prompts包括prompt_in和prompt_true
            attention_mask = torch.BoolTensor(attention_mask)  # 这里attention_mask对应pre_ids
            assert token_ids.shape[0]==attention_mask.shape[0]==prompts.shape[0]
            #print(token_ids.shape, token_types.shape, prompts.shape, attention_mask.shape)

            
        # pre_ids:[max_len-1],attention_mask:[max_len-1],label_ids:[max_len-1\len_label],prompt:[max_len-1\len_label,2,2]
        out =  input_tensor, token_ids, token_types, prompts, attention_mask
        if self.dummy:
            self.example_data = out
        return out
    
    def get_source_data(self, idx):
        with self.timers('get_sample'):
            sample = self.dataset[idx]  
        
        if sample is None:
            # if sample is broken choose another randomly
            return self.get_source_data(random.randint(0, self.dataset_length - 1))
        return sample
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
        """
        # SciPDFDataset.getitem(): return {"image": img, "prompt":data.pop("prompt"),"label": data.pop("label"),"pretext":data.pop("pretext"), "meta": data}
        # 全文prompt，len(pretext)=len(prompt)，len(label)=0
        # 非全文prompt，len(pretext) > len(prompt)= len(label)
        # data = self.dataset.__getitem__(idx)
        if self.dummy and self.example_data is not None:
            return self.example_data
        

        out = None
        fail_times=0
        while out is None and fail_times < 10:
            try:
                sample = self.get_source_data(idx)
           
                out = self.format_data(sample)
            except:
                pass
            idx = random.randint(0, self.dataset_length - 1)
            fail_times+=1
        if out is None:
            raise NotImplementedError
        return out
        

 
    def add_random_noise(self, boxes, attention_mask=None):
        '''
        add random noise to box,
        normal random with standard_variance=0.8*box sidelength
        '''
        bs,seq_len = boxes.shape[0],boxes.shape[1]
        boxes      = boxes.reshape(bs,seq_len,2,2)   # [bs,len,2,2] -> [bs*seq_len,2,2]
        output     = torch.zeros_like(boxes)
        if attention_mask is None:
            attention_mask_1 = (boxes == 0).all(dim=(-2,-1))
            attention_mask_2 = (boxes ==-1).any(dim=(-2,-1))
            attention_mask = attention_mask_1 | attention_mask_2
            attention_mask = ~attention_mask
            ### valid_mask = torch.unique(torch.where(torch.diff(boxes.reshape(-1,4)))[0]) # 除 [0,0,0,0](pad)或[-1,-1,-1,-1](mask)之外的box <== no [-1, -1, -1, -1] case anymore 
        boxes = boxes[attention_mask]
        assert boxes.ndim==3 # (B, 2,2)
        noise_scale = self.config.random_noise_scale 
        box_width   = boxes[:,1,0]-boxes[:,0,0]
        box_height  = boxes[:,1,1]-boxes[:,0,1]
        boxes[:,0,0] += torch.min(torch.normal(0.0,box_width*noise_scale) ,  box_width*noise_scale )   # x1
        boxes[:,0,1] += torch.min(torch.normal(0.0,box_height*noise_scale),  box_height*noise_scale)   # y1
        boxes[:,1,0] += torch.max(torch.normal(0.0,box_width*noise_scale) , -box_width*noise_scale )   # x2
        boxes[:,1,1] += torch.max(torch.normal(0.0,box_height*noise_scale), -box_height*noise_scale)   # y2
        boxes = torch.clamp(boxes,0,1)
        output[attention_mask] = boxes
        return output
        


    def format_training_data(self, token_ids:torch.LongTensor,
                                   token_types:torch.LongTensor,
                                   prompts:torch.FloatTensor ,
                                   attention_mask:torch.BoolTensor):
        input_token_ids      = token_ids[:,:-1]
        input_token_types    = token_types[:,:-1]
        input_prompts        = prompts[:,:-1]
        
        input_attention_mask = attention_mask[:,:-1]
        
        label_token_ids      = token_ids[:,1:]
        label_token_types    = token_types[:,1:]
        label_prompts        = prompts[:,1:]
        if self.training and self.config.random_noise_scale > 0: # this is a training process
            input_prompts = self.add_random_noise(input_prompts.clone(), input_attention_mask)
        return input_token_ids, input_token_types, input_prompts, input_attention_mask, label_token_ids, label_token_types, label_prompts


"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import logging
import os
from math import prod
from pathlib import Path
from typing import Dict, Tuple, Callable, Optional
from PIL import Image, UnidentifiedImageError

import torch
import json
import orjson
from torch.utils.data import Dataset
import numpy as np
from typing import List
import pandas as pd
import fitz
import io, json
from tqdm.auto import tqdm
import torch
from PIL import Image, ImageOps
fitz.TOOLS.mupdf_display_errors(on=False)
from .resource_utils import *
class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for processing a list of images using a preparation function.

    This dataset takes a list of image paths and applies a preparation function to each image.

    Args:
        img_list (list): List of image paths.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        img_list (list): List of image paths.
        prepare (Callable): The preparation function.
    """

    def __init__(self, img_list, prepare: Callable):
        super().__init__()
        self.img_list = img_list
        self.prepare = prepare

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            batch = [x for x in batch if x is not None and x[0] is not None]
            if len(batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(batch)
        except AttributeError:
            pass

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            return self.prepare(img)
        except:
            return


import numpy as np
class SciPDFDataset(Dataset):
    """
    Custom dataset for scientific PDF data.

    This dataset loads data from JSONL files and provides access to images, ground truth,
    and metadata.

    Args:
        path_to_index (str): Path to the index file.
        split (str, optional): Split of the dataset (e.g., "train", "test"). Default is "train".
        root_name (str, optional): Root directory name. Default is an empty string.
        template (str, optional): Template for split naming. Default is "%s".

    Attributes:
        empty_sample: Placeholder for empty samples.
    """

    empty_sample = None

    def __init__(
        self,
        path_to_index: str,
        split: str = "train",
        root_name="",
        template="%s",
    ) -> None:
        super().__init__()
        self.path_to_index = Path(path_to_index)    # path_to_index: .jsonl的path
        self.root_name = root_name
        self.path_to_root = self.path_to_index.parent
        self.split = split
        if not split in self.path_to_index.stem:    # split=validation, file=validation.jsonl
            pti = self.path_to_index.with_stem(self.path_to_index.stem.replace('train','validation'))
            if pti.exists():
                self.path_to_index = pti
            else:
                raise ValueError(f'Dataset file for split "{split}" not found: {pti}')
        self.dataset_file = None  # mulitprocessing
        # load seek map
        seek_path = self.path_to_root / (self.path_to_index.stem + ".seek.map") # 和jsonl同名、同目录的.seek.map文件
        if seek_path.exists():
            self.seek_map = orjson.loads(seek_path.open().read())
        else:
            raise ValueError(
                'No "%s" found in %s' % (seek_path.name, str(self.path_to_root))
            )
        self.dataset_length = len(self.seek_map)
        self.valided_index  = []
        self.baded_record   = set()
        self.good_record    = set()
    
    def __len__(self) -> int:
        return self.dataset_length

    def retreive_data(self, index: int) -> Optional[Dict]:
        position = self.seek_map[index]
        with open(self.path_to_index,'r') as fi:
            self.dataset_file = fi
            self.dataset_file.seek(position)
            line = self.dataset_file.readline()
        try:
            data: Dict = json.loads(line)
        except Exception as e:
            print(f"JSONL for sample {index} could not be loaded at position {position}: {str(e)}\n{line[:100]}")
            return self.empty_sample
        img = []
        img_path: Path = self.path_to_root / self.root_name / data.pop("image")
        if not img_path.exists():
            print(f"Sample {img_path} could not be found.")
            return self.empty_sample
        try:
            img = Image.open(img_path)
        except UnidentifiedImageError:
            print(f"Image {img_path} could not be opened.")
            return self.empty_sample
        
        out =  {"image": img, "prompt":data.pop("prompt"),"label": data.pop("label"),"pretext":data.pop("pretext"), "meta": data}
        # for key,val in out.items():
        #     print(f'key={key} => type={type(val)} => shape={np.array(val).shape}')
        # raise
        return out

    def choose_new_good_index(self):
        if len(self.valided_index) < 32:
            index = np.random.randint(0, self.dataset_length)
        else:
            index = np.random.choice(self.valided_index)
        return index

    def __getitem__(self, index: int) -> Optional[Dict]:
        if index in self.baded_record:
            index = self.choose_new_good_index()
        try_times = 0
        data = self.retreive_data(index)
        while data is None and try_times < 10 and self.split == 'train':
            self.baded_record.add(index)
            index = self.choose_new_good_index()
            data  = self.retreive_data(index)
        if data is None:
            raise ValueError(f"What a terrible dataset that I always get bad data")
        
        if index not in self.good_record:
            self.valided_index.append(index)
            self.good_record.add(index)

        return data


    def __iter__(self):
        for i in range(self.dataset_length):
            yield self[i]

split_alias  = {'train':'Train','validation':'Valid','test':'Test'}
split_alias2 = {'train':'train','validation':'valid','test':'test'}

class LougatDataset(Dataset):
    """
    Custom dataset for scientific PDF data.

    This dataset loads data from JSONL files and provides access to images, ground truth,
    and metadata.

    Args:
        path_to_index (str): Path to the index file.
        split (str, optional): Split of the dataset (e.g., "train", "test"). Default is "train".
        root_name (str, optional): Root directory name. Default is an empty string.
        template (str, optional): Template for split naming. Default is "%s".

    Attributes:
        empty_sample: Placeholder for empty samples.
    """

    empty_sample = None

    def __init__(self, path_to_index: str|pd.DataFrame|List[str], split: str = "train", root_name="", template="%s", random_dpi = False, auto_fix=False, tokenizer=None) -> None:
        super().__init__()
        if isinstance(path_to_index,str):
            path_to_index = pd.read_csv(path_to_index) ## (split:train, arxiv_path:0807/0807.2584, pdf_name: Index_Enhancement_Experiment, page_id: 8)
        elif isinstance(path_to_index, list):
            path_to_index = pd.concat([pd.read_csv(t) for t in path_to_index])
        assert isinstance(path_to_index, pd.DataFrame)
        self.path_to_index  = path_to_index[path_to_index['split']==split_alias[split.lower()]]
        if len(self.path_to_index)==0:
            self.path_to_index  = path_to_index[path_to_index['split']==split_alias2[split.lower()]]
        self.root_name      = root_name
        self.split          = split
        self.dataset_length = len(self.path_to_index)
        self.valided_index  = []
        self.baded_record   = set()
        self.good_record    = set()
        self.random_dpi     = random_dpi
        self.auto_fix       = auto_fix
        self.tokenizer = tokenizer
        self.empty_bbox_stratagy = 'follow_last'
        

    def __len__(self) -> int:
        return self.dataset_length

    def read_image_from_pdf(self, pdf_file_path, page_id):
        cache_path = pdf_file_path.replace('.colorful.text_only.pdf', f'.page_{page_id}.png').replace('boxed_pdf_image/', f'boxed_pdf_image/images/')
        if not self.random_dpi and os.path.exists(cache_path):
            img = Image.open(cache_path)
        else:
            dpi= 200
            if self.random_dpi:
                dpi = np.random.randint(100, 350)
            
            with fitz.open(pdf_file_path) as pdf:
                page = pdf.load_page(page_id)
                image = Image.open(io.BytesIO(page.get_pixmap(colorspace=fitz.csGRAY,dpi=dpi).pil_tobytes(format="PNG")))
                gray_image = ImageOps.grayscale(image)
                #img   = ImageOps.autocontrast(gray_image, cutoff=1)
                #img = gray_image.point(lambda p: 255 if p == 255 else 0) ### <-- this one let all no-white color become black
        return img
    

    def fill_empty_bbox(self, bbox_ordered):
        if self.empty_bbox_stratagy == "follow_last":
            return fill_empty_bbox_follow_last(bbox_ordered)
        else:
            raise NotImplementedError
    
    def read_text_bbox_from_csv(self, boxed_information_path):
        #cache_path = boxed_information_path.replace('text_bbox', f'text_cache').replace('.csv','.json')
        # if os.path.exists(cache_path):
        #     try:
        #         with open(cache_path,'r') as fi:
        #             data = json.load(fi)
        #             return None, data['bboxes'], data['text_type'], data['token_ids']
        #     except:
        #         pass # go native load
            
        df         = robust_csv_reader(boxed_information_path)
        status, df = filter_df(df)
        if not status:
            return None
        formated_token_bbox_pair = obtain_cleaned_token_type_bbox_pair(df)
        pretext, token_ids, bboxes, text_type = obtain_clean_tokenid_bbox_pair(formated_token_bbox_pair,self.tokenizer)
        
        return pretext, bboxes, text_type, token_ids



    def retreive_data(self, index: int) -> Optional[Dict]:
        row = self.path_to_index.iloc[index]
        arxivi_path=row['arxivpath']
        
        pdf_name=row['pdf_name']
        if pdf_name.endswith(".colorful.pdf"): 
            pdf_name = pdf_name[:-len(".colorful.pdf")]
        page_id =int(row['page_id'])

        pdf_name = pdf_name + '.colorful.text_only.pdf' if not pdf_name.endswith('.colorful.text_only.pdf') else pdf_name
        pdf_file_path = os.path.join(self.root_name, arxivi_path, 'boxed_pdf_image', pdf_name)
        img = self.read_image_from_pdf(pdf_file_path, page_id)
        boxed_information_path = os.path.join(self.root_name, arxivi_path, 'boxed_pdf_image', 'text_bbox_json',f"page_{page_id}.json")
        if not os.path.exists(boxed_information_path):
            boxed_information_path = os.path.join(self.root_name, arxivi_path, 'boxed_pdf_image', 'text_bbox',f"page_{page_id}.csv")
        reading_result = self.read_text_bbox_from_csv(boxed_information_path)
        if reading_result is None:return None
        pretext, bboxes, token_type, token = reading_result
        bboxes = self.fill_empty_bbox(bboxes)
        bboxes = np.array(bboxes)
        bboxes = np.clip(bboxes, 0, 1)
        metadata={'path': pdf_file_path, 'box_path': boxed_information_path}
        out= {"image": img, "prompt":bboxes,"token_type": token_type,"pretext":pretext, "meta": metadata, "token": token}
        if len(out["pretext"])>4096:
            tqdm.write(f"PDF for sample {self.path_to_index.iloc[index]['arxivpath']+'/'+self.path_to_index.iloc[index]['pdf_name']}.page{self.path_to_index.iloc[index]['page_id']} exceed")
        # for key,val in out.items():
        #     print(f'key={key} => type={type(val)} => shape={np.array(val).shape}')
        # raise

        return out

    def choose_new_good_index(self):
        if len(self.valided_index) < 32:
            index = np.random.randint(0, self.dataset_length)
        else:
            index = np.random.choice(self.valided_index)
        return index

    def __getitem__(self, index: int) -> Optional[Dict]:
        if index in self.baded_record:
            index = self.choose_new_good_index()
        
        data = self.retreive_data(index)
        try_times = 0
        limit = 10
        while data is None and try_times < limit : 
            try:
                data = self.retreive_data(index)
            except:
                tqdm.write(f"PDF for sample {self.path_to_index.iloc[index]['arxivpath']+'/'+self.path_to_index.iloc[index]['pdf_name']}.page{self.path_to_index.iloc[index]['page_id']} could not be loaded")
                self.baded_record.add(index)
                index = self.choose_new_good_index()
            try_times +=1 
            assert try_times<limit, f"Can't get good data for index {index}"
        if data is None:
            raise ValueError(f"What a terrible dataset that I always get bad data")
        
        if index not in self.good_record:
            self.valided_index.append(index)
            self.good_record.add(index)

        return data


    def __iter__(self):
        for i in range(self.dataset_length):
            yield self[i]


# coding=utf-8
# Copyright 2024 Microsoft and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Florence-2.
"""

from transformers import CLIPImageProcessor
import re
import logging
from typing import List, Optional, Union
import numpy as np

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType
from transformers import AutoProcessor, VisionEncoderDecoderConfig

logger = logging.getLogger(__name__)

# Copied from transformers.models.idefics2.processing_idefics2.is_url


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")

# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)

from transformers.models.nougat import NougatProcessor

class Florence2Processor(ProcessorMixin):
    r"""
    Constructs a Florence2 processor which wraps a Florence2 image processor and a Florence2 tokenizer into a single processor.

    [`Florence2Processor`] offers all the functionalities of [`CLIPImageProcessor`] and [`BartTokenizerFast`]. See the
    [`~Florence2Processor.__call__`] and [`~Florence2Processor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`NougatTokenizer`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError(
                "Image processor is missing an `image_seq_length` attribute.")
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor


    def _construct_prompts(self, text):
        # replace the task tokens with the task prompts if task token is in the text
        prompts = []
        for _text in text:
            # 1. fixed task prompts without additional inputs
            for task_token, task_prompt in self.task_prompts_without_inputs.items():
                if task_token in _text:
                    assert _text == task_token, f"Task token {task_token} should be the only token in the text."
                    _text = task_prompt
                    break
            # 2. task prompts with additional inputs
            for task_token, task_prompt in self.task_prompts_with_input.items():
                if task_token in _text:
                    _text = task_prompt.format(
                        input=_text.replace(task_token, ''))
                    break
            prompts.append(_text)
        return prompts

    def __call__(
        self,
        images=None,
        text=None,
        return_tensors: Optional[Union[str, TensorType]] = 'np',
        do_resize: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional["ChannelDimension"] = "channels_first",  # noqa: F821
        input_data_format: Optional[
            Union[str, "ChannelDimension"]  # noqa: F821
        ] = None,
        resample: "PILImageResampling" = None,  # noqa: F821
        do_convert_rgb: bool = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_rescale: bool = None,
    ) -> BatchFeature:
        

        return_token_type_ids = False

        if images is not None:
            inputs = self.image_processor(
                images,
                do_resize=do_resize,
                do_normalize=do_normalize,
                return_tensors=return_tensors,
                image_mean=image_mean,
                image_std=image_std,
                input_data_format=input_data_format,
                data_format=data_format,
                resample=resample,
                do_convert_rgb=do_convert_rgb,
                # do_thumbnail = do_thumbnail,
                # do_align_long_axis = do_align_long_axis,
                do_rescale = do_rescale,
            )
        if text is not None:
            encodings = None
            raise NotImplementedError

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs
 

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Florence2
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Florence2
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

AutoProcessor.register(VisionEncoderDecoderConfig,Florence2Processor)
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

from transformers import CLIPImageProcessor, SamImageProcessor
import logging
from typing import List, Optional, Union
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType
from transformers import AutoProcessor, PretrainedConfig

logger = logging.getLogger(__name__)


class UparxiveAdvanceProcessor(ProcessorMixin):
    r"""
    Constructs a Florence2 processor which wraps a Florence2 image processor and a Florence2 tokenizer into a single processor.

    [`UparxiveAdvanceProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`BartTokenizerFast`]. See the
    [`~UparxiveAdvanceProcessor.__call__`] and [`~UparxiveAdvanceProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`NougatTokenizer`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor" ### <-- this one must be a existed image processor in transformer
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
        
        super().__init__(image_processor, tokenizer)
        self.image_processor = self.image_processor


    def __call__(
        self,
        images=None,
        return_tensors: Optional[Union[str, TensorType]] = 'np',
        do_resize: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional["ChannelDimension"] = "channels_first",  # noqa: F821
        input_data_format: Optional[Union[str, "ChannelDimension"]] = None,
        resample: "PILImageResampling" = None,  # noqa: F821
        do_convert_rgb: bool = None,
        do_rescale: bool = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
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
                do_thumbnail = do_thumbnail,
                do_align_long_axis = do_align_long_axis,
                do_rescale = do_rescale,
            )
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

#UparxiveAdvanceProcessor.register_for_auto_class()
AutoProcessor.register(PretrainedConfig,UparxiveAdvanceProcessor)
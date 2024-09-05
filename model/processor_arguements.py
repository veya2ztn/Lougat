
from dataclasses import dataclass,field,fields

from transformers import AutoImageProcessor, ProcessorMixin
from transformers import NougatProcessor, NougatImageProcessor
from .decoder.locr.image_processing_locr import LougatImageProcessor ## <--make sure the Image processing work
from .image_processing_uparxive import UparxiveImageProcessor
from .processing_uparxive import *
@dataclass
class ProcessorConfig:
    processor_fold: str
    train_processor_fold: str = None
    _processor = {}

    def get_processor(self, mode='eval')->NougatProcessor:
        if mode not in self._processor:
            if mode == 'train':
                processor_fold = self.train_processor_fold
            else:
                processor_fold = self.processor_fold
            self._processor[mode] = self.create_processor(processor_fold)
        return self._processor[mode]

    @staticmethod
    def create_processor(processor_fold:str):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(processor_fold,trust_remote_code=True)

        return processor    

    def to_fields_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
@dataclass
class NougatProcessorConfig(ProcessorConfig):
    processor_fold:str = field(default='config/processor/nougat')
    


@dataclass
class LougatProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/lougat')



@dataclass
class FlougatProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/flougat')
@dataclass
class FlougatXProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/flougatX')

@dataclass
class UparxiveProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/uparxive')


@dataclass
class UparxiveBetaProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/uparxive_beta')

@dataclass
class Uparxive1KProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/uparxive1k')



@dataclass
class UparxiveBetaProcessor2kConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/uparxive_beta_2k')


@dataclass
class UparxiveProcessor2kConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/uparxive2k')


@dataclass
class UparxiveXProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/uparxiveX')

@dataclass
class SlougatProcessorConfig(ProcessorConfig):
    processor_fold: str = field(default='config/processor/slougat')


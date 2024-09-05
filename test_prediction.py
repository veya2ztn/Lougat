
from predictor.PromptPredictor import *
from accelerate import Accelerator
from torch.utils.data import DataLoader
import sys
model_path = sys.argv[1] # 'checkpoints/Lougat/lorc_small/lougat_bfloat16_flash_attn_more/best/weight/epoch0100'
class DummyDataset:
    def __init__(self,*args,**kargs):
        pass
    def __len__(self):
        return 10000
config    = PredictorConfig(pretrain_path=model_path, input_path='demo/1512.03385.pdf', output_path='output')
print(config)
accelerator   = Accelerator()
#config.processer_path = "config/processor/flougat" #config.pretrain_path
config.processer_path = config.pretrain_path
config.dtype =  torch.bfloat16 if accelerator.state.mixed_precision == "bf16" else torch.float32
print(config.dtype )
predictor = PromptPredictor(config)
inputs = config.input_path
if isinstance(inputs, str):
    inputs = Path(inputs)
elif isinstance(inputs, list):
    inputs = [Path(i) for i in inputs]


train_dataset = DummyDataset()
datasetloader = DataLoader(train_dataset, batch_size=1)
optimizer = torch.optim.Adam(predictor.model.parameters(), lr=1e-3)
predictor.model,datasetloader,optimizer = accelerator.prepare(predictor.model,datasetloader,optimizer)
predictor.predict(inputs, config.output_path, True)

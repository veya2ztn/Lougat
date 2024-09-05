import torch
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import re
from .predict_arguement import PredictorConfig
from model.decoder.locr.modeling_locr import LougatForVision2Seq
from model.decoder.locr.image_processing_locr import LougatImageProcessor
from model.decoder.llougat.modeling_llougat import LlougatForVision2Seq
from model.decoder.llougat.processing_florence2 import Florence2Processor
from transformers import AutoProcessor, AutoModel
from typing import List,Tuple
#from nougat.dataset.rasterize import rasterize_paper
from dataset.rasterize import rasterize_paper
import os

from transformers import NougatProcessor

class PromptPredictor:
    
    def __init__(self, config:PredictorConfig):
        self.config   = config
        self.processor= AutoProcessor.from_pretrained(config.processer_path,trust_remote_code=True)

        self.model    = AutoModel.from_pretrained(config.pretrain_path,torch_dtype=config.dtype).cuda()
        #self.model.set_position_decay(1)

    @staticmethod
    def get_whole_pdf_images(pdf):
        if isinstance(pdf, (Path,str)):
            pdf = [pdf]
        pdf_images = []
        for pdf_path in pdf:
            pdf_path = Path(pdf_path)
            name = os.path.basename(pdf_path)
            if str(pdf_path).endswith('.pdf'):
                pages = rasterize_paper(pdf_path)
                for page in pages:
                    image = Image.open(page)
                    pdf_images.append([image, 0])
                pdf_images[-1][1] =1
            else:
                pdf_images.append([Image.open(pdf_path),1])
        return pdf_images

    def turn_ids_to_str(self, ids):
        sequence = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        sequence = self.processor.tokenizer.post_process_generation(sequence, fix_markdown=True)
        return sequence
    
    def predict(self, pdf:Path|List[Path]|Tuple[Path], output_dir,recompute):
        model      = self.model
        device     = self.model.device 
        processor  = self.processor
        get_whole_pdf_images = self.get_whole_pdf_images(pdf)
        predictions = []
        scores = []
        file_index = 0
        page_num = 0
        pdf_error = False
        print("=====================")
        
        for i, (sample, is_last_page) in enumerate(tqdm(get_whole_pdf_images)):
            sample = self.processor(sample, return_tensors="pt").pixel_values
            # batch = torch.load("input.llougat.pt")
            # batch = {k:v[:1] for k,v in batch.items()}
            # sample   = batch['image_tensors']
            # input_ids= batch['pre_input_ids']
            # label_id = batch['label_id']
            # prompt_in= batch["prompt_in"]
            # model_output = model.generate(sample.to(device,self.config.dtype), 
            #                               start_tokens = None,
            #                               start_bboxes = prompt_in[:,1:2],
            #                               min_length=1, 
            #                               max_length=10,
            #                               bad_words_ids=[[processor.tokenizer.unk_token_id]])

            model_output = model.generate(sample.to(device,self.config.dtype),
                                          min_length=1, 
                                          max_length=1000,
                                          bad_words_ids=[[processor.tokenizer.unk_token_id]])
            pred_tokens = model_output['sequences'][0] #self.processor.tokenizer.convert_ids_to_tokens(model_output['sequences'][0])
            #pred_bboxes = model_output['prompt_bbox'][0].view(-1, 4)
            #real_tokens = input_ids[0,: len(pred_tokens)]
            # iou = bbox_iou(true_bbox,pred_bbox)
            # true_bbox = ",".join([f"{int(t*255):03d}" for t in true_bbox])
            # pred_bbox = ",".join([f"{int(t*255):03d}" for t in pred_bbox])
            
            #real_bboxes = prompt_in[0, : len(pred_tokens)].view(-1, 4)

            # for pred_token, true_token, pred_bbox, true_bbox in zip(pred_tokens,real_tokens, pred_bboxes,real_bboxes):
            #     pred_bbox = ",".join([f"{int(t*255):03d}" for t in pred_bbox])
            #     true_bbox = ",".join([f"{int(t*255):03d}" for t in true_bbox])
            #     print(f"Token: [{true_token}]<->[{pred_token}]\n      |{true_bbox} <=>\n      |{pred_bbox}")
            # raise
            #model_output["repetitions"] = self.turn_ids_to_str(model_output["repetitions"])
            model_output["predictions"] = self.turn_ids_to_str(model_output["sequences"])
            #print(model_output["predictions"])
            output = model_output["predictions"]
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                pdf_error  = True
            
            elif "repeats" in model_output and model_output["repeats"] is not None and model_output["repeats"][0] is not None:
                # if model_output["repeats"][j] > 0:
                # If we end up here, it means the output is most likely not complete and was truncated.
                predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                predictions.append(output)
                predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                pdf_error = True
            else:
                predictions.append(output)

            if True or is_last_page: # 一页以输出：if True:
                # one pdf file compeleted, clear the predictions and pdf_error
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                print(out)
                if output_dir:   
                    if pdf_error:
                        out_path = output_dir / Path('error') / Path(pdf).with_suffix(".mmd").name
                    else:
                        out_path = output_dir / Path('correct') / Path(pdf).with_suffix(".mmd").name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(out, encoding="utf-8")
                    # perplexity = perplexity if perplexity else np.inf
                else:
                    print(out, "\n\n")
                predictions = []
                scores = []
                page_num = 0
                file_index += 1
                pdf_error = False
            break

if __name__ == '__main__':
    config    = PredictorConfig(pretrain_path='debug/20240309', input_path='demo/1512.03385.pdf', output_path='output')
    predictor = PromptPredictor(config)
    inputs = config.input_path
    if isinstance(inputs, str):
        inputs = Path(inputs)
    elif isinstance(inputs, list):
        inputs = [Path(i) for i in inputs]
    predictor.predict(inputs, config.output_path, False)

from __future__ import annotations

from typing import Any, Callable, Sequence, Sized

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.utils import Timers

import time,os
import numpy as np
from transformers.trainer_pt_utils import distributed_broadcast_scalars
import json
from accelerate.utils import DistributedType
from .utils import DummyProgressBar, DistributedTqdmProgressBar, LossTracker, DataSimfetcher
from .train_arguements import TrainConfig, get_parallel_config_of_accelerator
from accelerate.data_loader import DataLoaderShard, DataLoaderDispatcher
import accelerate
from utils import print0
import traceback
# from nougat.metrics import get_metrics

class SpeedTestFinishError(NotImplementedError):pass

class Trainer:
    def __init__(
        self,
        *,
        model,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader | None ,
        optimizer: Optimizer,
        lr_scheduler,
        accelerator: Accelerator,
        train_config: TrainConfig,
        epoch_end_callbacks: Sequence[Callable[['Trainer'], None]] | None = None,
    ):
        self.train_config = train_config
        self.model = model
        
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.not_valid_during_train = train_config.not_valid_during_train
        self.epochs = train_config.epochs
        self.log_interval = train_config.Monitor.log_interval
        self.save_on_epoch_end = train_config.save_on_epoch_end
        self.clip_value = train_config.clip_value
        
        self.train_loss_tracker      = LossTracker()
        self.validation_loss_tracker = LossTracker()
        # if isinstance(self.train_dataloader.dataset, Sized):
        #     num_steps_per_epoch = len(self.train_dataloader)
        # else:
        #     num_steps_per_epoch = None
        num_steps_per_epoch = len(self.train_dataloader)
        self.progress_bar = DistributedTqdmProgressBar(self.epochs, num_steps_per_epoch=num_steps_per_epoch,
                                                       #bar_log_path=os.path.join(self.accelerator.project_dir,'log',f'train_bar.log'), 
                                                       desc="Training....",)
        self.valid_progress_bar = DistributedTqdmProgressBar(self.epochs, num_steps_per_epoch=len(self.validation_dataloader), 
                                                    desc="Validating....",)
        self.epoch_end_callbacks = epoch_end_callbacks or []
        self.current_step = 0
        #self.timers = Timers()
        self.best_validation_loss = np.inf
        self.best_validation_metrics = {}
        weight_dir   = os.path.join(self.accelerator.project_dir, 'best', 'weight')
        os.makedirs(weight_dir, exist_ok=True)
        self.cached_weight_dir = weight_dir
        self.metric_report_path = os.path.join(self.accelerator.project_dir,'metric_report.json')
           
    def set_train_mode(self, mode='train'):
        
        self.model.train()
        model = self.accelerator.unwrap_model(self.model)
        model.freeze_model_during_train(self.train_config.Freeze)

    def compute_full_loss(self,error_pool):
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

    def log_speed_test_end_exit(self,):
        if self.accelerator.is_main_process:
            current_rate = self.progress_bar.progress_bar.format_dict['rate']
            current_speed= current_rate*self.batch_size ## batch_size assign out of trainer
            if os.path.exists('speed_test.json'):
                with open('speed_test.json', 'r') as f: 
                    speed_record = json.load(f)
            else:
                speed_record = {}
            current_processing = self.accelerator.num_processes
            parallel_config    = get_parallel_config_of_accelerator(self.accelerator)
            distributed_type = parallel_config['distributed_type']
            model_type = self.model_name
            if model_type not in speed_record:
                speed_record[model_type]={}
            if distributed_type not in speed_record[model_type]:
                speed_record[model_type][distributed_type]={}
            if current_processing not in speed_record[model_type][distributed_type]:
                speed_record[model_type][distributed_type][current_processing]={}
            if self.batch_size not in speed_record[model_type][distributed_type][current_processing]:
                speed_record[model_type][distributed_type][current_processing][self.batch_size]=[]
            data_type = parallel_config['_mixed_precision'] if distributed_type is not DistributedType.DEEPSPEED else "bf16"
            speed_record[model_type][distributed_type][current_processing][self.batch_size].append([data_type, current_speed])
            setting = f"SET:{model_type}|{distributed_type}.{data_type}.G{current_processing}.B{self.batch_size}"
            print(f"{setting} => {current_speed}")
            os.system(f"""echo "{setting} => {current_speed}" >> speed_test.record.log """)
            with open('speed_test.json', 'w') as f: 
                    json.dump(speed_record, f)
        raise SpeedTestFinishError

    def format_training_data(self,batch):
        image_tensors, token_ids ,token_types,prompts ,attention_mask = batch
        (input_token_ids, input_token_types, input_prompts, 
         input_attention_mask, 
         label_token_ids, label_token_types, label_prompts) = self.train_dataloader.dataset.format_training_data(token_ids ,token_types,prompts ,attention_mask)
        batch =  {
                "image_tensors":image_tensors,
                "pre_input_ids":input_token_ids,
                "attention_mask":input_attention_mask,
                "label_id":label_token_ids,
                "prompt_in":input_prompts,
                "prompt_true":label_prompts,
                #"input_token_types":input_token_types,
                "label_token_types":label_token_types,
        }

        batch = self.auto_precision(batch,['pre_input_ids','label_id','prompt_in',"full_prompt_in"])

        return batch
    
    def train(self, start_epoch=0):
        validation_loss=None
        update_intervel = max(len(self.train_dataloader)//1000,2)
        nan_count = 0
        error_tracker = {}
        for current_epoch in range(0, self.epochs + 1):
            if current_epoch < start_epoch:
                self.current_step+=len(self.train_dataloader)
                continue
            if self.lr_scheduler is not None and current_epoch>1:self.lr_scheduler.step(current_epoch-1)
            if not self.train_config.do_validation_at_first_epoch and current_epoch==0:continue
            self.train_dataloader.dataset.train()
            if current_epoch >= 1:
                self.set_train_mode()
                self.progress_bar.on_epoch_start(current_epoch)
                data_loading = []
                model_train = []
                #self.timers('data_loading').start()
                #featcher = DataSimfetcher(self.train_dataloader,device=self.accelerator.device)
                last_record_time = time.time()
                #failed_times=  0
                
                # for batch_index in range(len(self.train_dataloader)):
                #     batch = featcher.next()
                if not isinstance(self.train_dataloader, (DataLoaderShard,DataLoaderDispatcher)):
                    self.train_dataloader.sampler.set_epoch(current_epoch)
                for batch_index, batch in enumerate(self.train_dataloader):
                    # if batch_index == 0: print(f"{self.accelerator.process_index}=>{batch['idx']}")
                    #if batch_index > 10:break
                    # continue
                    batch = self.format_training_data(batch)

                    data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
                    #<-- In accelerate mode, the time record will be quite large and the model time will be quite small
                    #<-- However, the totally cost remain, thus it is a typical bug inside accelerate.
                    
                    error_record={}
                    with self.accelerator.accumulate(self.model):
                        self.optimizer.zero_grad()
                        loss, error_pool = self.model(**batch)[0:2] 
                        #loss          = self.compute_full_loss(error_pool)
                        error_record = {k:v.item() for k,v in error_pool.items() if v is not None}
                        #print(error_pool)      
                        if not torch.isnan(loss):
                            nan_count = 0
                        else:
                            nan_count+=1 
                            assert nan_count < 10, f"too many nan detected, exit"
                        self.train_loss_tracker.update(loss) 
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients and self.clip_value:
                            self.accelerator.clip_grad_value_(self.model.parameters(), self.clip_value)
                        self.optimizer.step()

                        
                    model_train.append(time.time() - last_record_time);last_record_time =time.time()
                    
                    if batch_index%update_intervel==1:self.progress_bar.update(update_intervel)
                    self.current_step += 1
                     
                    if batch_index % self.log_interval == 0 or batch_index < 30:
                        log_dict = {'loss': self.train_loss_tracker.loss,
                                    'data': np.mean(data_loading),
                                    'model': np.mean(model_train),
                                    #'time':self.timers.get_string()
                                    }|error_record
                        show_dict = dict([[k,v] for k,v in log_dict.items() if "e_" not in k])          
                            
                        self.progress_bar.show_metrics(show_dict)
                        for key, val in error_pool.items():
                            if val is None:continue
                            if key not in error_tracker:error_tracker[key] = LossTracker()
                            error_tracker[key].update(val)
                        error_metric = {k:t.loss for k,t in error_tracker.items()}
                        train_metrics = self.add_prefix(log_dict|error_metric, 'itering')
                        self.accelerator.log(train_metrics, step=self.current_step)
                        data_loading = []
                        model_train = []

                    last_record_time = time.time()
                        
                    #self.timers('data_loading').start()
                    if self.train_config.time_test and batch_index>50:break
                self.accelerator.wait_for_everyone()

                if self.train_config.time_test:
                    self.log_speed_test_end_exit()
                for key, val in error_pool.items():
                    if val is None:continue
                    if key not in error_tracker:error_tracker[key] = LossTracker()
                    error_tracker[key].update(val)
                error_metric = {k:t.loss for k,t in error_tracker.items()}
                train_metrics = self.add_prefix({'loss': self.train_loss_tracker.loss, 
                                                 'lr':self.optimizer.param_groups[0]['lr']
                                                 }|error_metric, 
                                                 'train')
                
                train_metrics['epoch'] = current_epoch
                self.accelerator.log(train_metrics, step=self.current_step)
                #self.accelerator.log(1)
                self.train_loss_tracker.on_epoch_end()
                self.progress_bar.on_epoch_end()
                try:
                    self.save_state(current_epoch)
                except:
                    traceback.print_exc()
                    #print("Failed to save the state")
                    pass
                    
            

            if self.validation_dataloader and not self.not_valid_during_train:
                
                validation_loss=None
                error_pool = {}
                ######### collect validation infomration #########
                error_pool = self.evaluate(
                    self.validation_dataloader,
                    self.validation_loss_tracker
                )
                error_pool = self._nested_gather_scalar_dict(error_pool)
                # print(error_pool)
                # for key, val in error_pool.items():
                #     if val is None:continue
                #     print(f"{key:20s} => {val:.3f}")
                #     raise
                if 'loss' in error_pool:
                    validation_loss  = error_pool['loss'] 
                else:
                    key = list(error_pool.keys())[0]
                    validation_loss  = error_pool[key]

                validation_metrics_pool = error_pool
                validation_metrics = self.add_prefix(validation_metrics_pool, 'validation')
                validation_metric_string = "\n".join([f'    {key}: {val:.4f}' for key, val in validation_metrics.items()])
                self.accelerator.print(f'Epoch {current_epoch}:\n{validation_metric_string}')
                
                
                validation_metrics['epoch'] = current_epoch
                self.accelerator.log(validation_metrics, step=self.current_step)
                ######### save the best weight #########
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.update_best_state_information(validation_metrics,current_epoch)
                    self.maintain_the_cached_weight(current_epoch)
                
            

            if self.epoch_end_callbacks:
                for callback in self.epoch_end_callbacks:
                    callback(self)
            self.accelerator.wait_for_everyone()
            
            #torch.cuda.empty_cache()
            
        #self.accelerator.log(self.add_prefix(self.best_validation_metrics, 'best'))
        
        
        if self.train_config.clean_checkpoints_at_end and self.accelerator.is_main_process:
            the_checkpoints_path = os.path.join(self.accelerator.project_dir, 'checkpoints', f'checkpoints_{self.epochs-1}')
            os.system(f'rm -rf {the_checkpoints_path}/*')
            os.system(f'touch {the_checkpoints_path}/cleanup_due_to_finish_train')

    def save_state(self,epoch):
        if epoch % self.train_config.Checkpoint.save_every_epoch != 0:return
        if not self.save_on_epoch_end:return 
        self.accelerator.project_configuration.iteration = epoch + 1 # <-- align whole the iteration <--- semem will save checkpoint every epoch
        # Should enable all process for DEEPSPEED: https://github.com/huggingface/diffusers/issues/2606
        self.accelerator.save_state()
        # if self.accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM] or int(accelerate.__version__.split('.')[-2])<24:
        #     self.accelerator.save_state()
        # else:
        #    if self.accelerator.is_main_process:
        #         self.accelerator.save_state()
        return
    
    def _nested_gather_scalar_dict(self, _dict):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        new_dict = _dict
        if _dict is None:
            return 
        #### try to sync the distributed tensor and compute the correct error cross GPU. Fail! ####
        if (self.accelerator.distributed_type != DistributedType.NO):
            new_dict = {}
            for key, scalar in _dict.items():
                new_dict[key]=torch.mean(distributed_broadcast_scalars([scalar])).item()
        return new_dict

    def get_metric_report(self):
        metric_report_path = self.metric_report_path 
        if os.path.exists(metric_report_path):
            with open(metric_report_path, 'r') as f: 
                now_metric_report = json.load(f)
        else:
            now_metric_report = {}
        return now_metric_report
    
    def update_metric_report(self, key, val, epoch, weight_path):
        metric_report_path = self.metric_report_path 
        now_metric_report = self.get_metric_report()
        now_metric_report[key]   = {
            'score':val,
            'epoch':epoch,
            'path':weight_path
        }
        with open(metric_report_path, 'w') as f:
            json.dump(now_metric_report, f, indent=4)

    def auto_precision(self, torch_pool,except_key=[]):
        if self.accelerator.state.mixed_precision !='bf16':return torch_pool
        if self.accelerator.distributed_type not in [DistributedType.FSDP, DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:return torch_pool
        dtype = torch.bfloat16 if self.accelerator.state.mixed_precision == "bf16" else torch.float32
        for key,val in torch_pool.items():
            if key in except_key:continue
            if isinstance(val,torch.Tensor):
                if isinstance(val, torch.LongTensor):
                    torch_pool[key] = val
                else:
                    torch_pool[key] = val.to(dtype=dtype)
            elif isinstance(val,dict):
                torch_pool[key] = self.auto_precision(val)
            elif isinstance(val,list):
                torch_pool[key] = [self.auto_precision(v) for v in val]
            elif isinstance(val,tuple):
                torch_pool[key] = tuple([self.auto_precision(v) for v in val])
        return torch_pool
        
        
    def update_best_state_information(self, validation_metrics: dict, current_epoch: int):
        for key in validation_metrics.keys():
            if key in ['epoch']:continue
            if "loss_" in key: continue
            if 'realtime' in key:continue
            if 'posttime' in key:continue # do not save the metric like validation.posttime.s.ws3.t0.recall_beta.at0.5
            if 'record' in key: continue
            self.update_best_state_by_key(
                key, validation_metrics, current_epoch)
    
    def update_best_state_by_key(self, key: str, validation_metrics: dict, current_epoch: int):
        large_is_better = (key.split('/')[-1][0]=='a') or ('precision' in key) or ('recall' in key) or ('iou' in key and 'diou' not in key)
        small_is_better = not large_is_better
        
        if key not in validation_metrics:
            raise ValueError(f'Key {key} not found in validation metrics')
        validation_loss      = validation_metrics[key]
        if key not in self.best_validation_metrics:
            self.best_validation_metrics[key]=np.inf if small_is_better else 0
        best_validation_loss = self.best_validation_metrics[key]
        goodQ = (validation_loss < best_validation_loss) if small_is_better else (validation_loss > best_validation_loss)
        if (validation_loss and  goodQ):
            save_dir = os.path.join(self.accelerator.project_dir, 'best', key.strip('/').replace('/','.'))
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'score'), 'a') as f:f.writelines(f"{key} {current_epoch} {validation_loss}\n")
            linked_checkpoint_path = self.get_current_cached_weight_path(current_epoch)
            self.update_metric_report(key, self.best_validation_metrics[key], 
                                      current_epoch, linked_checkpoint_path)
            self.create_soft_link(linked_checkpoint_path, os.path.join(save_dir, 'checkpoint'))
        if small_is_better:
            self.best_validation_metrics[key] = min(validation_loss, best_validation_loss)
        else:
            self.best_validation_metrics[key] = max(validation_loss, best_validation_loss)

    def create_soft_link(self, linked_checkpoint_path, softlink_of_checkpoints):
        #linked_checkpoint_path = "checkpoints/stead.trace.BDLEELSSO_1000of3000/ViT_MS_A.3.512.1024.256.4/10_11_17_21-seed_21/best/weight/epoch-7.bin"
        #softlink_of_checkpoints = "checkpoints/stead.trace.BDLEELSSO_1000of3000/ViT_MS_A.3.512.1024.256.4/10_11_17_21-seed_21/best/validation.loss_x/checkpoint.bin"
        relative_path = os.path.relpath(linked_checkpoint_path,os.path.dirname(softlink_of_checkpoints))
        os.system(f"rm {softlink_of_checkpoints}; ln -s {relative_path} {softlink_of_checkpoints}")
        
        #print(softlink_of_checkpoints)
        #os.symlink(relative_path,softlink_of_checkpoints)
    
    def get_current_cached_weight_path(self, current_epoch):
        return os.path.join(self.cached_weight_dir,f'epoch{current_epoch:04d}')

    def get_used_checkpoint_path(self):
        now_metric_report = self.get_metric_report()
        return [pool['path'] for pool in now_metric_report.values()]
    
    def maintain_the_cached_weight(self, current_epoch):
        if current_epoch < 1: return
        existed_checkpoint_path = [os.path.join(self.cached_weight_dir, p) for p in os.listdir(self.cached_weight_dir)]
        used_checkpoint_path    = self.get_used_checkpoint_path()
        now_checkpoint_path     = self.get_current_cached_weight_path(current_epoch)
        if now_checkpoint_path in used_checkpoint_path:
            if now_checkpoint_path not in existed_checkpoint_path:
                unwrapper_model = self.model
                while hasattr(unwrapper_model,'module'):
                    unwrapper_model = unwrapper_model.module
                unwrapper_model.save_pretrained(now_checkpoint_path,safe_serialization=False)
                self.validation_dataloader.dataset.processor.save_pretrained(now_checkpoint_path)
        for path in existed_checkpoint_path:
            if path not in used_checkpoint_path:
                ## clean the unused checkpoint
                print(f"clean the unused checkpoint {path}")
                os.system(f'rm -r {path}')

    def log_metrics(self, metrics, step: int, flag='iter'):

        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_tracker: LossTracker | None = None
    ):

        update_intervel = 10
        self.model.eval()
        
        dataloader.dataset.eval() 
        error_tracker= {}        
        unwrapper_model = self.accelerator.unwrap_model(self.model)
        self.valid_progress_bar.on_epoch_start()
        for batch_index,batch in enumerate(dataloader):
            if batch is None:continue
            with torch.inference_mode():
                batch = self.format_training_data(batch)
                loss, metrics = self.model(**batch)[0:2]
                
                metrics['loss'] = loss
                for key, val in metrics.items():
                    if val is None:continue
                    if key not in error_tracker:error_tracker[key] = LossTracker()
                    error_tracker[key].update(val)
            if batch_index%update_intervel==1:
                self.valid_progress_bar.update(update_intervel)
            #if batch_index > 10:break
            #if self.train_config.do_validation_at_first_epoch and batch_index > 10:break
        self.valid_progress_bar.on_epoch_end()
        out = dict([[k,t.loss] for k,t in error_tracker.items()])
        return out

    def benchmark(
        self,
        dataloader: DataLoader,
        loss_tracker: LossTracker | None = None,
        generate_eval: bool = False
    ):

        update_intervel = 10
        self.model.eval()
        
        dataloader.dataset.eval() 
        error_tracker= {}        
        unwrapper_model = self.accelerator.unwrap_model(self.model)
        self.valid_progress_bar.on_epoch_start()
        for batch_index,batch in enumerate(dataloader):
            if batch is None:continue
            with torch.inference_mode():
                image_tensors,pre_input_ids, attention_masks, label_ids, prompts = batch
                
                batch =  {
                            "image_tensors":image_tensors,
                            "input_ids":pre_input_ids,
                            "attention_mask":attention_masks,
                            "prompt":prompts
                    }
                batch  = self.auto_precision(batch,except_key=['pre_input_ids','label_id','prompt','prompt_in'])
                output = unwrapper_model.inference(return_attentions=True,validation=True,**batch)
                gts    = unwrapper_model.processor.batch_decode(label_ids, skip_special_tokens=True) # label token  
                preds  = output["predictions"]
                torch.save(gts, 'gts.pt')                   
                torch.save(preds, 'preds.pt')    
                metrics = get_metrics(gts, preds)

                for key, val in metrics.items():
                    if key not in error_tracker:error_tracker[key] = LossTracker()
                    error_tracker[key].update(val)
            #if batch_index%update_intervel==1:
            self.valid_progress_bar.update(update_intervel)
            #if batch_index > 10:break
            #if self.train_config.do_validation_at_first_epoch and batch_index > 10:break
        self.valid_progress_bar.on_epoch_end()
        out = dict([[k,t.loss] for k,t in error_tracker.items()])
        return out
    

    def test_inference(self):
        if os.path.exists("input.pt"):
            batch = torch.load("input.pt")
        else:
            for batch_index,batch in enumerate(self.validation_dataloader):
                break
            batch = self.format_training_data(batch)
        batch = {k:v[:1] for k,v in batch.items() }
        batch = self.auto_precision(batch,['pre_input_ids','label_id','prompt_in',"full_prompt_in"])
        attention_mask = batch["attention_mask"].bool()
        batch_in = {k:v for k,v in batch.items() if k in ['prompt_in','pre_input_ids','image_tensors','attention_mask']}
        batch_in['prompt_in'] = batch_in['prompt_in'][attention_mask][None]
        batch_in['pre_input_ids'] = batch_in['pre_input_ids'][attention_mask][None]
        batch_in['attention_mask'] = None
        with torch.inference_mode():
            loss, metrics, decoder_outputs = self.model(**batch_in)
        for key, val in metrics.items():
            if val is None:continue
            print(f"{key:20s} => {val.item():.3f}")
        logits, preded_bbox = decoder_outputs[:2]
        preded_bbox = preded_bbox[0] if isinstance(preded_bbox, tuple) else preded_bbox
        if batch_in['attention_mask'] is not None:
            batch_in['attention_mask'] = batch_in['attention_mask'].bool()
            logits = logits[batch_in['attention_mask']] 
            preded_bbox = preded_bbox[batch_in['attention_mask']].view(-1, 4)
        else:
            logits = logits[0]
            preded_bbox = preded_bbox[0].view(-1, 4)
        labels = batch["label_id"]
        labels = labels[attention_mask]
        
        tokenid     = torch.argmax(logits, dim=-1) # (BL, 50000)-> (B,L)
        
        # preded_bbox -> (B, L, 2, 2)
        prompt_true = batch["prompt_true"]
        prompt_true = prompt_true[attention_mask].view(-1, 4)
        
        pred_bboxes = preded_bbox.view(-1,4).float().detach().cpu().numpy()
        real_bboxes = prompt_true.view(-1,4).float().detach().cpu().numpy()
        pred_tokens = tokenid.float().detach().cpu().numpy()
        real_tokens = labels.float().detach().cpu().numpy()
        

        result_check = {}
        result_check['true_tokens'] = real_tokens
        result_check['true_bboxes'] = real_bboxes

        # token_result= torch.stack([labels,tokenid],-1).float().detach().cpu().numpy()
        # bbox_result = torch.stack([prompt_true, preded_bbox], -2).float().detach().cpu().numpy()
        print("=========== inference result =================")
        ioues = [bbox_iou(true_bbox,pred_bbox) for pred_bbox,true_bbox in zip(pred_bboxes,real_bboxes)]
        result_check['bbox_iou_inference'] = ioues
        result_check['pred_tokens_inference'] = pred_tokens
        result_check['pred_bboxes_inference'] = pred_bboxes
        for iou,pred_token, true_token, pred_bbox, true_bbox in zip(ioues[:10], pred_tokens,real_tokens, pred_bboxes,real_bboxes):
            pred_bbox = ",".join([f"{int(t*255):03d}" for t in pred_bbox])
            true_bbox = ",".join([f"{int(t*255):03d}" for t in true_bbox])
            #print(f"Token: [{true_token}]<->[{pred_token}]\n      |{true_bbox} <=>\n      |{pred_bbox}")
            print(f"IOU:{iou:.2f}| {true_bbox} <=> [{int(true_token)}] \n        | {pred_bbox}     [{int(pred_token)}]")
        # for i in range(10):
        #     true_bbox, pred_bbox = bbox_result[i]
        #     true_token,pred_token= token_result[i]
        #     iou = bbox_iou(true_bbox,pred_bbox)
        #     true_bbox = ",".join([f"{int(t*255):03d}" for t in true_bbox])
        #     pred_bbox = ",".join([f"{int(t*255):03d}" for t in pred_bbox])
        #     print(f"IOU:{iou:.2f}| {true_bbox} <=> [{int(true_token)}] \n        | {pred_bbox}     [{int(pred_token)}]")
        
        
        

        print("=========== generate result =================")
        with torch.inference_mode():
            generate_output = self.model.generate(image_tensors = batch['image_tensors'], max_length=1000)
        pred_bboxes = generate_output['prompt_bbox'][0].view(-1,4).float().detach().cpu().numpy()
        pred_tokens = generate_output['sequences'][0].float().detach().cpu().numpy()
        real_tokens = batch['pre_input_ids'][0,: len(pred_tokens)].float().detach().cpu().numpy()
        real_bboxes = batch['prompt_in'][0, : len(pred_tokens)].view(-1, 4).float().detach().cpu().numpy()

        ioues = [bbox_iou(true_bbox,pred_bbox) for pred_bbox,true_bbox in zip(pred_bboxes,real_bboxes)]
        result_check['bbox_iou_generate'] = ioues
        result_check['pred_tokens_generate'] = pred_tokens
        result_check['pred_bboxes_generate'] = pred_bboxes

        for iou,pred_token, true_token, pred_bbox, true_bbox in zip(ioues[:10], pred_tokens,real_tokens, pred_bboxes,real_bboxes):
            pred_bbox = ",".join([f"{int(t*255):03d}" for t in pred_bbox])
            true_bbox = ",".join([f"{int(t*255):03d}" for t in true_bbox])
            #print(f"Token: [{true_token}]<->[{pred_token}]\n      |{true_bbox} <=>\n      |{pred_bbox}")
            print(f"IOU:{iou:.2f}| {true_bbox} <=> [{int(true_token)}] \n        | {pred_bbox}     [{int(pred_token)}]")

        print("=========== generate result once sign start =================")
        with torch.inference_mode():
            generate_output = self.model.generate(image_tensors = batch['image_tensors'], 
                                                  start_tokens = None,
                                                  start_bboxes = batch_in['prompt_in'][:,1:2],
                                                  max_length=1000)
        pred_bboxes = generate_output['prompt_bbox'][0].view(-1,4).float().detach().cpu().numpy()
        pred_tokens = generate_output['sequences'][0].float().detach().cpu().numpy()
        real_tokens = batch['pre_input_ids'][0,: len(pred_tokens)].float().detach().cpu().numpy()
        real_bboxes = batch['prompt_in'][0, : len(pred_tokens)].view(-1, 4).float().detach().cpu().numpy()
        ioues = [bbox_iou(true_bbox,pred_bbox) for pred_bbox,true_bbox in zip(pred_bboxes,real_bboxes)]
        result_check['bbox_iou_generate2'] = ioues
        result_check['pred_tokens_generate2'] = pred_tokens
        result_check['pred_bboxes_generate2'] = pred_bboxes

        for iou,pred_token, true_token, pred_bbox, true_bbox in zip(ioues[:10], pred_tokens,real_tokens, pred_bboxes,real_bboxes):
            pred_bbox = ",".join([f"{int(t*255):03d}" for t in pred_bbox])
            true_bbox = ",".join([f"{int(t*255):03d}" for t in true_bbox])
            #print(f"Token: [{true_token}]<->[{pred_token}]\n      |{true_bbox} <=>\n      |{pred_bbox}")
            print(f"IOU:{iou:.2f}| {true_bbox} <=> [{int(true_token)}] \n        | {pred_bbox}     [{int(pred_token)}]")

        result_check["image_tensors"]= batch['image_tensors']
        savedir = os.path.dirname(self.train_config.Checkpoint.preload_weight)
        torch.save(result_check, os.path.join(savedir, "the_inference_generate_test_result.pt"))
        self.visulize_test(result_check)
        
    def visulize_test(self, generate_result):
        min_length = min(len(generate_result[k]) for k in ['pred_tokens_generate',
                                                   'pred_tokens_inference',
                                                   'pred_tokens_generate',
                                                   'pred_tokens_generate2'])-1
        import matplotlib.pyplot as plt
        fig,axes=  plt.subplots(3,1,figsize=(16,4))
        for ax,key in zip(axes,['pred_tokens_inference', 'pred_tokens_generate', 'pred_tokens_generate2']):
            start = 0 if key == 'pred_tokens_inference' else 1 
            x = np.arange(min_length)
            y = (generate_result[key][start:start+min_length] - generate_result['true_tokens'][:min_length])==0
            ax.scatter(x,y,label=key)
            ax.set_title(key)
        savedir = os.path.dirname(self.train_config.Checkpoint.preload_weight)
        plt.tight_layout()
        fig.savefig(os.path.join(savedir, "token_predition.png") )

        fig,ax=  plt.subplots(figsize=(16,9))
        for key in ['bbox_iou_inference', 'bbox_iou_generate', 'bbox_iou_generate2']:
            ax.plot(generate_result[key][:750],label=key)
        ax.legend()
        fig.savefig(os.path.join(savedir, "bbox_predition.png") )


    def turn_ids_to_str(self, ids):
        sequence = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        sequence = self.processor.post_process_generation(sequence, fix_markdown=True)
        return sequence
    
    def quick_test(self):
        from PIL import Image
        import re
        from transformers import AutoProcessor
        model      = self.model
        model.eval()
        device     = self.model.device 
        processor  = self.processor = AutoProcessor.from_pretrained("config/processor/test")
        get_whole_pdf_images = [Image.open("demo/0704.0017_1.png")]
        predictions = []
        scores = []
        file_index = 0
        page_num = 0
        pdf_error = False

        for i, sample in enumerate(tqdm(get_whole_pdf_images)):
            sample = processor(sample, return_tensors="pt").pixel_values
            model_output = model.generate(sample.to(device), min_length=1,
                                          bad_words_ids=[[processor.tokenizer.unk_token_id]],only_return_sequences=False)
            model_output["repetitions"] = self.turn_ids_to_str(model_output["repetitions"])
            model_output["predictions"] = self.turn_ids_to_str(model_output["sequences"])
            #print(model_output["predictions"])
            output = model_output["predictions"]
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                pdf_error  = True
            
            elif model_output["repeats"] is not None and model_output["repeats"][0] is not None:
                # if model_output["repeats"][j] > 0:
                # If we end up here, it means the output is most likely not complete and was truncated.
                predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                predictions.append(output)
                predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                pdf_error = True
            else:
                predictions.append(output)

            out = "".join(predictions).strip()
            out = re.sub(r"\n{3,}", "\n\n", out).strip()
            print(out)
            break

import numpy as np

def bbox_iou(pred, target, epsilon=1e-5):
    '''
    args: 
    pred/target: [bs, length, 4]
    '''
    pred  = np.array(pred)
    target= np.array(target)

    # Coordinates of the intersection box
    inter_x1 = np.maximum(pred[ 0], target[ 0])
    inter_y1 = np.maximum(pred[ 1], target[ 1])
    inter_x2 = np.minimum(pred[ 2], target[ 2])
    inter_y2 = np.minimum(pred[ 3], target[ 3])
    
    # Ensure intersection area is not negative
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    
    # Area of the predicted and target boxes
    pred_area = (pred[ 2] - pred[ 0]) * (pred[ 3] - pred[ 1])
    target_area = (target[ 2] - target[ 0]) * (target[ 3] - target[ 1])
    
    # Union area
    union_area = pred_area + target_area - inter_area
    
    # IoU calculation
    iou = inter_area / (union_area + epsilon)
    
    return iou


from typing import List
import Levenshtein as lev
def get_metrics(gt: List[str], pred: List[str]):
    distances = [lev.distance(p, g) for p, g in zip(pred, gt)]
    return {'distance': np.mean(distances)}
import os, wandb, json
import torch
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.tracking import TensorBoardTracker, WandBTracker

from train.optimizer.build_optimizer import get_optimizer
from train.scheduler.build_scheduler import get_scheduler
from dataset.loader import load_data
from model import load_model
from model.utils import getModelSize

from utils import get_rank, get_local_rank, smart_load_weight, is_ancestor, printg, print0, think_up_a_unique_name_for_inference,smart_read_weight_path
from config.utils import print_namespace_tree, flatten_dict, retrieve_dict
from dataset.dataset_arguements import DataloaderConfig
from train.train_arguements import TrainConfig
from project_arguements import ProjectConfig, get_args, to_dict
from evaluator.evaluator_arguements import EvaluatorConfig, EvalPlotConfig
from train.train_arguements import get_parallel_config_of_accelerator

import albumentations as alb
import logging
from torch.utils.data import DataLoader
# Set up basic configuration for logging
# This will only display WARNING and above logs
logging.basicConfig(level=logging.WARNING)
os.environ['WANDB_CONSOLE']='off'
def load_optimizer_and_schedule(model, train_dataloader, args: TrainConfig):
    optimizer = get_optimizer(model, args.Optimizer)
    scheduler = get_scheduler(optimizer, train_dataloader, args.Scheduler)    
    return optimizer, scheduler


def load_dataloader(args:DataloaderConfig, infer=False, needed_dataset = None, test_dataloader=False):
    
    local_rank = get_local_rank(args)
    if needed_dataset is None:
        needed=['train', 'valid'] if not infer  else ['valid']
    else:
        needed= [t.lower() for t in needed_dataset.split(',')]
    dataset_pool = load_data(args.Dataset, needed=needed,test_dataloader=test_dataloader)
    train_dataset, valid_dataset, test_dataset = dataset_pool['train'], dataset_pool['valid'], dataset_pool['test']
    #print(sampling_strategy)
    if train_dataset is not None:
        print0(f"================> Train dataset length: {len(train_dataset)} <================")
    if valid_dataset is not None:
        print0(f"================> Valid dataset length: {len(valid_dataset)} <================")
    if test_dataset  is not None: 
        print0(f"================> TEST dataset length: {len(test_dataset)} <================")
    #train_dataset.return_idx = True
    ####train_dataset.return_idx = True
    if infer:
        raise NotImplementedError
        for dataset in [train_dataset, valid_dataset, test_dataset]:
            if dataset is None: continue
            dataset.return_idx = True


    train_dataloader = valid_dataloader = test_dataloader = None
    if args.donot_use_accelerate_dataloader:
        from torch.utils.data.distributed import DistributedSampler
        assert args.data_parallel_dispatch or not args.multi_gpu
        # if args.multi_gpu:printg(f"""
        #     WARNING: if you use the native multi-gpu dataloader, you need manually call dataloader.sampler.set_epoch(epoch) in each epoch. 
        #             [Currently(20231124), We dont realize such feature.]
        #     """)
        num_workers = args.num_workers
        if train_dataset is not None:
            train_datasampler = DistributedSampler(train_dataset, shuffle=True) if args.multi_gpu else None
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_datasampler,
                                        num_workers=num_workers,
                                        pin_memory=not args.not_pin_memory, drop_last=True if not infer else False)
        if valid_dataset is not None:
            valid_datasampler = DistributedSampler(valid_dataset, shuffle=False) if args.multi_gpu else None
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_datasampler,
                                        num_workers=num_workers,
                                        pin_memory=not args.not_pin_memory, drop_last=False,)
        if test_dataset is not None:
            test_datasampler = DistributedSampler(test_dataset, shuffle=False) if args.multi_gpu else None
            test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_datasampler,
                                        num_workers=num_workers,
                                        pin_memory=not args.not_pin_memory, drop_last=False,)
    else:
        num_workers = args.num_workers if args.data_parallel_dispatch or local_rank == 0 else 0
        if train_dataset is not None:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=num_workers,
                                        pin_memory=not args.not_pin_memory, drop_last=True if not infer else False)
        if valid_dataset is not None:
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        pin_memory=not args.not_pin_memory, drop_last=False,)
        if test_dataset is not None:
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        pin_memory=not args.not_pin_memory, drop_last=False,)
    return train_dataloader, valid_dataloader, test_dataloader

import time
def build_accelerator(args: ProjectConfig):
    if accelerate.__version__ in ['0.24.0', '0.24.1']:
        printg(f"""
            WARNING:accelerate version {accelerate.__version__} has a bug that will not random shuffle the dataloader. Please downgrade to 0.23.0.
            See https://github.com/huggingface/accelerate/issues/2157 """)
        exit(0)
    use_wandb = (isinstance(args.task, TrainConfig) and args.task.Monitor.use_wandb)
    if get_rank()!=0:
        time.sleep(1)
    os.makedirs(args.output_dir, exist_ok=True)
    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir),
        automatic_checkpoint_naming=True,
        total_limit=args.task.Checkpoint.num_max_checkpoints,
    )

    
    log_with = []
    if isinstance(args.task, TrainConfig): log_with += ['tensorboard']
    if use_wandb:log_with.append("wandb")
    if len(log_with)==0:log_with=None
    aacelerator_config = {
        'dataloader_config':accelerate.DataLoaderConfiguration(dispatch_batches=not args.DataLoader.data_parallel_dispatch),
        'project_config': project_config,
        'log_with': log_with
    }
    if isinstance(args.task, TrainConfig):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.task.find_unused_parameters)
        aacelerator_config['kwargs_handlers'] = [ddp_kwargs]
        aacelerator_config['gradient_accumulation_steps'] = args.task.gradient_accumulation_steps
        set_seed(args.task.seed)
        
    
    accelerator = Accelerator(**aacelerator_config)  # in accelerate>0.20.0, the dispatch logic is changed. thus in low drive and low gpu, it will stuck or raise after reinitialize the datalodaer .See https://github.com/OpenAccess-AI-Collective/axolotl/issues/494
    
    

    
    accelerator.init_trackers(
        project_name=f"{args.dataset_name}",
        config=None,
        init_kwargs={"wandb": {'group': args.model_name, 'name': args.trial_name, 'settings': wandb.Settings(_disable_stats=True)} } if use_wandb else {}
    )

    
    if accelerator.is_main_process:
        cfg = to_dict(args)
  
        cfg['parallel_config'] = get_parallel_config_of_accelerator(accelerator)
        cfg = retrieve_dict(cfg,exclude_key=['downstream_pool','parallel_config'])
        if 'trained_batch_size' not in cfg or cfg['trained_batch_size'] is None or isinstance(args.task,TrainConfig):
            trained_batch_size = cfg['parallel_config']['num_processes']*cfg['batch_size'] * cfg.get('gradient_accumulation_steps',1)
            cfg['trained_batch_size'] = trained_batch_size
            args.model.trained_batch_size = trained_batch_size                                                                 
        for tracker in accelerator.trackers:
            if isinstance(tracker, TensorBoardTracker):
                pool = flatten_dict(cfg)
                board_pool = {}
                for key, val in pool.items():
                    if isinstance(val, list):
                        val = ",".join([str(t) for t in val])
                    board_pool[key] = val 
                tracker.store_init_configuration(board_pool)
            elif isinstance(tracker, WandBTracker):
                tracker.store_init_configuration(cfg)
            else:
                tracker.store_init_configuration(cfg)
    
    
    # if use_wandb and accelerator.is_main_process:
    #     accelerator.trackers[-1].store_init_configuration()
    

    accelerator.print(f'Output dir: {args.output_dir}')


    if accelerator.is_main_process:
        print_namespace_tree(args)
        if isinstance(args.task, TrainConfig):
            config_path = os.path.join(args.output_dir, 'train_config.json')
            #save(args, path=config_path, save_dc_types=True, indent=4)
            with open(config_path, 'w') as f:
                #json.dump(convert_namespace_tree(args), f, indent=4)
                #print(retrieve_dict(to_dict(args)))
                #raise
                json.dump(retrieve_dict(to_dict(args)), f, indent=4)

    args.DataLoader.multi_gpu = accelerator.state._shared_state['backend'] is not None
    #print(args.DataLoader.multi_gpu)
    if accelerator.is_local_main_process:
        print(f"Current Node: {os.uname().nodename}")
    return accelerator


from train.utils import DummyProgressBar, DistributedTqdmProgressBar
from train.PromptTrainer import Trainer
from tqdm.auto import tqdm
def main(args: ProjectConfig):
    save_on_epoch_end   = True
    epoch_end_callbacks = None
    
    accelerator = build_accelerator(args)
    # if isinstance(args.task, EvalPlotConfig):
    #     save_root    = args.task.plot_data_dir ## direct use the /visualize path
    #     if accelerator.is_main_process:plot_evaluate(args, save_root)
    #     return 
    
    

    needed_dataset = args.task.eval_dataflag if isinstance(args.task, EvaluatorConfig) else None
    # needed_dataset = 'train,valid,test'
    train_dataloader, valid_dataloader, test_dataloader = load_dataloader(args.DataLoader, 
                                                                          infer=isinstance(args.task, EvaluatorConfig),
                                                                          needed_dataset=needed_dataset, test_dataloader=args.test_dataloader)
    if args.test_dataloader:
        train_dataloader, valid_dataloader = accelerator.prepare(train_dataloader, valid_dataloader)
        if train_dataloader is not None:
            for i, data in enumerate(tqdm(train_dataloader)):
                # if i%100==1:
                #     train_dataloader.dataset.timers.log()
                pass
        if valid_dataloader is not None:
            for i, data in enumerate(tqdm(valid_dataloader)):
                pass
        if test_dataloader is not None:
            for i, data in enumerate(tqdm(test_dataloader)):
                pass
        return

    

    model = load_model(args.model)
    if isinstance(args.task, TrainConfig): model.freeze_model_during_train(args.task.Freeze)
    #model = torch.nn.Conv2d(3, 3, 3)
    #if isinstance(args.task, TrainConfig): model.freeze_model_during_train(args.task.Freeze)
    param_sum, buffer_sum, all_size = getModelSize(model)
    accelerator.print(f" Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB\n")
    
    optimizer, lr_scheduler = None, None
    if isinstance(args.task, TrainConfig): ### If use this, we can not claim an optimizer in the evaluator thus not support the Flash Attention BF16 for evaluation
        optimizer, lr_scheduler = load_optimizer_and_schedule(model, train_dataloader, args.task)     
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        lr_scheduler = None
    
    ## You must load weight before Deepspeed, otherwise it does not work
    if args.task.Checkpoint.preload_weight:
        printg(f"LOADING MODEL from {args.task.Checkpoint.preload_weight}")
        unwrapper_model = model
        while hasattr(unwrapper_model,'module'):
            unwrapper_model = unwrapper_model.module
        smart_load_weight(unwrapper_model, smart_read_weight_path(args.task.Checkpoint.preload_weight,device=accelerator.device),
                          strict=not args.task.Checkpoint.load_weight_partial, shape_strict=not args.task.Checkpoint.load_weight_ignore_shape)
        #Shape mismatching always throws an exceptions. Only key mismatching can be ignored.

    # torch.save(unwrapper_model.state_dict(), "pretrain_weights/slougat.matched_start.pt")
    # raise
    if hasattr(args.model.encoder, 'compile_image_encoder') and args.model.encoder.compile_image_encoder:
        model.encoder.sam_image_encoder = torch.compile(model.encoder.sam_image_encoder)

    #train_dataloader, valid_dataloader, optimizer, lr_scheduler, model = accelerator.prepare(train_dataloader, valid_dataloader, optimizer, lr_scheduler, model)\
    if args.DataLoader.donot_use_accelerate_dataloader:
        optimizer, lr_scheduler, model = accelerator.prepare(optimizer, lr_scheduler, model)
    else:
        train_dataloader, valid_dataloader, optimizer, lr_scheduler, model = accelerator.prepare(train_dataloader, valid_dataloader, optimizer, lr_scheduler, model)

    
    
    start_epoch = 0
    if args.task.Checkpoint.preload_state:
        print(f"resume from {args.task.Checkpoint.preload_state}")
        accelerator.load_state(args.task.Checkpoint.preload_state)
        if args.task.Checkpoint.continue_train:
            start_epoch = int(os.path.split(args.task.Checkpoint.preload_state)[-1].replace('checkpoint_',""))
            old_best_weight_path = os.path.join(os.path.dirname(os.path.dirname(args.task.Checkpoint.preload_state)), 'best')
            if os.path.exists(old_best_weight_path) and accelerator.is_main_process:
                new_best_weight_path = f"{old_best_weight_path}.epoch{start_epoch}"
                if not os.path.exists(new_best_weight_path):
                    print(f"rename {old_best_weight_path} to {new_best_weight_path}")
                    os.system(f"mv {old_best_weight_path} ")
                accelerator.project_configuration.iteration = start_epoch
        else:
            start_epoch = 0
            accelerator.project_configuration.iteration = start_epoch


    if args.DataLoader.loader_all_data_in_memory_once:
        raise NotImplementedError(f"we dont allow all data into memory since it is too large")  


    if isinstance(args.task, EvaluatorConfig):
        raise NotImplementedError
        assert args.task.Checkpoint.preload_weight is not None
        assert is_ancestor(args.output_dir, args.task.Checkpoint.preload_weight), f"the output_dir {args.output_dir} is not the ancestor of preload_state {args.task.Checkpoint.preload_state}"
        branch=args.task.eval_dataflag
        accelerator.print(f"Notice!!!! You are testing on branch ======> {branch} <======")
        infer_mode   = args.task.infer_mode
        if branch in ['TRAIN']:
            dataloader = train_dataloader
        elif branch in ['DEV','VALID']:
            dataloader = valid_dataloader
        elif branch in ['TEST']:
            dataloader = test_dataloader 
        else:
            raise NotImplementedError
        
        infer_result = get_evaluate_detail(model, dataloader, args.task)
        
        save_root    = os.path.join(args.output_dir, 'visualize',branch)
        if not os.path.exists(save_root):os.makedirs(save_root,exist_ok=True)
        data_name = think_up_a_unique_name_for_inference(args)
        save_data_root    = os.path.join(save_root, f'{data_name}_data')
        if not os.path.exists(save_data_root):os.makedirs(save_data_root,exist_ok=True)
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        data_save_root = os.path.join(save_data_root, f'infer_result_GPU{local_rank}')
        printg(f"save visual data in {data_save_root}")
        accelerator.wait_for_everyone()
        save_dict_of_numpy(infer_result, data_save_root)
        if accelerator.is_main_process: 
            with open(os.path.join(save_data_root, 'infer_config.json'), 'w') as f:
                json.dump(retrieve_dict(to_dict(args)), f, indent=4)
            #save(args, path=os.path.join(save_data_root, 'infer_config.yaml'), indent=4)
        #torch.save(infer_result,os.path.join(save_data_root, f'infer_result.GPU{local_rank}.pt'))
        #return
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process: 
            plot_evaluate(args, save_data_root)
            if args.task.clean_up_plotdata:os.system(f"rm -r {save_data_root}")
        accelerator.wait_for_everyone()
        return 

    
    if accelerator.is_main_process and args.task.Monitor.wandbwatch:
        wandb.watch(model, log_freq = args.task.Monitor.wandbwatch)
    #torch.cuda.empty_cache()
    # Trainer
    
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        train_config=args.task,
    )
    if args.quick_debug:
        trainer.quick_test()
        return
    if args.quick_test_inference:
        trainer.test_inference()
        return
    
    print(f"======> GPU:{accelerator.process_index} is ready for training..........")
    accelerator.print(f'Start training for totally {args.task.epochs} epochs')
    
    trainer.model_name = args.model.nickname
    trainer.batch_size = args.DataLoader.batch_size
    trainer.train(start_epoch)
    
    #accelerator.wait_for_everyone()
    accelerator.print('Training finished')
    
    if accelerator.is_main_process:
        unwrapper_model = model
        while hasattr(unwrapper_model,'module'):
            unwrapper_model = unwrapper_model.module
        unwrapper_model.save_pretrained(args.output_dir,safe_serialization=False)
    #accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     os.system("""sleep 30; ps -axuf|grep wandb| awk '{print $2}'| xargs kill""")
    accelerator.end_training()
    
if __name__ == "__main__":
    args = get_args()
    #print(args)
    main(args)
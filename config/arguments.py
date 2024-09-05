import argparse
import os
import torch
import random 
import json



def build_parser():
    parser = argparse.ArgumentParser(
        description='Arguments', allow_abbrev=False, add_help=False)

    # Standard arguments.
    parser = _add_model_args(parser)
    parser = _add_training_args(parser)
    parser = _add_optimizer_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_dataset_args(parser)
    parser = _add_valid_args(parser)
    parser = _add_monitor_args(parser)
    parser = _add_single_station_training_args(parser)
    return parser

def get_plain_parser():
    parser = build_parser()
    return parser.parse_args()

def structure_args(args):
    
    new_args = argparse.Namespace(
        model=_structure_model_args(args),
        train=_structure_training_args(args),
        valid=_structure_valid_args(args),
        data=_structure_dataset_args(args),
        optimizer=_structure_optimizer_args(args),
        monitor=_structure_monitor_args(args),
        single_station_train=_structure_single_station_training_args(args),
        debug=args.debug
    )
    new_args.valid.pga = args.n_pga_targets > 0
    return new_args
def get_args_parser():
    """Parse all arguments in structure way"""
    args = get_plain_parser()
    args = structure_args(args)
    
    return args

def flatten_dict(_dict):
    out = {}
    for key, val in _dict.items():
        if isinstance(val, dict):
            for k, v in val.items():
                out[k] = v 
        else:
            out[key] = val
    return out
    
def get_args(config_path=None):
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--conf_file",  default=None, help="Specify config file", metavar="FILE")
    conf_parser.add_argument("-m", "--model_config",  default=None, help="Specify config file", metavar="FILE")
    conf_parser.add_argument("-t", "--train_config",  default=None, help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        'GFNet training and evaluation script', parents=[build_parser()])
    defaults = {}
    config_path = config_path if config_path else args.conf_file

    if config_path:
        with open(config_path, 'r') as f:defaults = json.load(f)
        
        if 'model_config' in defaults:
            args.model_config = defaults['model_config']
            print("given config has model config, overwrite other model_config")
        if 'train_config' in defaults:
            args.train_config = defaults['train_config']
            print("given config has train config, overwrite other train_config")
        new_pool = {}
        for key, val in defaults.items(): 
            # check the code below , there are many key and attr have different name, which means the json load cannot recovery exactly the origin configuration.
            # since the model_config and train_config maybe also get changed during develop.
            # load a trail from json config is experiment.
            if isinstance(val, dict):
                for k,v in val.items():new_pool[k] = v
            else:
                new_pool[key] = val
        parser.set_defaults(**flatten_dict(new_pool))

    
    if args.model_config:
        with open(args.model_config, 'r') as f:defaults = json.load(f)
        parser.set_defaults(**flatten_dict(defaults))

    
    
    if args.train_config:
        with open(args.train_config, 'r') as f:defaults = json.load(f)
        parser.set_defaults(**flatten_dict(defaults)) 
    
    
    config = parser.parse_known_args(remaining_argv)[0]
    config.config_file = args.conf_file
    config = structure_args(config)

    if args.model_config:config.model_config = args.model_config
    if args.train_config:config.train_config = args.train_config
    return config


##############################################
############### Model Setting ################
##############################################
def _add_model_args(parser):
    group = parser.add_argument_group(title='network size')

    # Global arguments
    group.add_argument("--model", type=str, default="transformer")
    group.add_argument("--model_type", type=str)
    group.add_argument("--max_stations", type=int, default=25)
    group.add_argument('--embedding_size', type=int, default=150,help='embedding_size')
    group.add_argument('--find_unused_parameters',
                       action='store_true', help='')
    # Single Station Setting prediction config
    group.add_argument("--single_mlp_dims"   , type=lambda s: tuple(map(int, s.split(","))), default="150,100,50,30")
    group.add_argument("--wave_downsample"   , type=int, default=5)

    # Magnitude prediction config
    group.add_argument("--mag_output_dim", type=int, default=10)
    group.add_argument("--mag_mlp_dims", type=lambda s: tuple(map(int, s.split(","))), default="150,100,50,30")
    group.add_argument("--mag_mixture", type=int, default=5)
    group.add_argument("--mag_bias_mu", type=float, default=1.8)
    group.add_argument("--mag_bias_sigma", type=float, default=0.2)

    # Location prediction config
    group.add_argument("--loc_output_dim", type=int, default=50)
    group.add_argument("--loc_mlp_dims", type=lambda s: tuple(map(int, s.split(","))), default="150,100,50,50")
    group.add_argument("--loc_mixture", type=int, default=5)
    group.add_argument("--loc_bias_mu", type=float, default=0)
    group.add_argument("--loc_bias_sigma", type=float, default=1)

    # Position embedding config
    group.add_argument("--wavelengths", type=lambda s: tuple((float(x), float(y)) for x, y in (item.split(":") for item in s.split(","))), default="0.01:10,0.01:10,0.01:10")
    group.add_argument("--borehole", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
    group.add_argument("--rotation", type=float, default=None)
    group.add_argument("--rotation_anchor", type=float, default=None)

    # Core engine config
    group.add_argument("--nhead", type=int, default=10)
    group.add_argument("--num_encoder_layers", type=int, default=6)
    group.add_argument("--num_decoder_layers", type=int, default=6)
    group.add_argument("--dim_feedforward", type=int, default=2048)
    group.add_argument("--dropout", type=float, default=0.0)
    group.add_argument("--batch_first", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True)

    # PGA targets config
    group.add_argument("--n_pga_targets", type=int, default=0)
    group.add_argument("--pga_mixture", type=int, default=5)
    group.add_argument("--pga_bias_mu", type=float, default=-5)
    group.add_argument("--pga_bias_sigma", type=float, default=1)

    # Event config
    group.add_argument("--event_token_init_range", type=float, default=None)
    group.add_argument("--no_event_token", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)

    # Dataset bias config
    group.add_argument("--n_datasets", type=int, default=0)

    # Global model params
    group.add_argument("--alternative_coords_embedding", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
    group.add_argument("--activation", type=str, default="relu")

    group.add_argument("--model_alias", type=str, default=None)

    group.add_argument("--only_train_latlon", action='store_true', help='more_epoch_train')
    group.add_argument("--simpleloss", action='store_true', help='more_epoch_train')
    
    return group
def _structure_model_args(args):
    single_prediction_config={
        "mlp_dims":args.single_mlp_dims,
        "downsample":args.wave_downsample,
        "channel": 6 if args.borehole else 3
    }

    magnitude_prediction_config={
        "output_dim":args.mag_output_dim,
        "mlp_dims":args.mag_mlp_dims,
        "mixture":args.mag_mixture,
        "bias_mu":args.mag_bias_mu,
        "bias_sigma":args.mag_bias_sigma
    }

    location_prediction_config={
        "output_dim":args.loc_output_dim,
        "mlp_dims":args.loc_mlp_dims,
        "mixture":args.loc_mixture,
        "bias_mu":args.loc_bias_mu,
        "bias_sigma":args.loc_bias_sigma
    }

    pga_targets_config={
        "n_pga_targets":args.n_pga_targets,
        "mixture": args.pga_mixture,
        "bias_mu":args.pga_bias_mu,
        "bias_sigma":args.pga_bias_sigma
    }

    position_embedding_config={
        "wavelengths":args.wavelengths,
        "borehole":args.borehole,
        "rotation":args.rotation,
        "rotation_anchor":args.rotation_anchor
    }

    core_engine_config={
        "nhead":args.nhead,
        "num_encoder_layers":args.num_encoder_layers,
        "num_decoder_layers":args.num_decoder_layers,
        "dim_feedforward":args.dim_feedforward,
        "dropout":args.dropout,
        "batch_first":args.batch_first
    }

    event_config={
        "event_token_init_range":args.event_token_init_range,
        "no_event_token":args.no_event_token
    }

    dataset_bias_config={
        "n_datasets":args.n_datasets
    }

    model_params = argparse.Namespace(
        model_type=args.model_type,
        embedding_size=args.embedding_size,
        waveform_model_config=single_prediction_config,
        magnitude_prediction_config=magnitude_prediction_config,
        location_prediction_config=location_prediction_config,
        pga_targets_config=pga_targets_config,
        position_embedding_config=position_embedding_config,
        core_engine_config=core_engine_config,
        event_config=event_config,
        dataset_bias_config=dataset_bias_config,
        alternative_coords_embedding=args.alternative_coords_embedding,
        activation=args.activation,
        max_stations=args.max_stations,
        find_unused_parameters=args.find_unused_parameters,
        model_alias=args.model_alias,
        simpleloss=args.simpleloss,
        only_train_latlon=args.only_train_latlon
    )

    return model_params

##############################################
############## Train Setting #################
##############################################
def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--seed', type=int, default=1234,help='Random seed used for python, numpy, ''pytorch, and cuda.')
    group.add_argument('--batch-size', type=int, default=2,help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--use_amp', action='store_true', help='whether continue train')
    group.add_argument('--continue_train', action='store_true', help='whether continue train')
    group.add_argument('--pretrain_weight', type=str, default=None,help='the pretrain weight path')
    parser.add_argument("--do_train_shuffle",action='store_true', help='whether continue train',default=True)
    group.add_argument('--epochs', type=int, default=10,help='epochs')
    group.add_argument('--job_mode', type=str, default='pretrain',help='mode')
    group.add_argument('--save_warm_up', type=int,
                       default=1, help="save_warm_up")
    group.add_argument('--save_every_epoch', type=int,
                       default=1, help="save_every_epoch")
    group.add_argument('--epoch_save_list', type=lambda s: tuple(map(int, s.split(","))), 
                       default="99")
    group.add_argument('--debug',action='store_true', help='whether continue train')
    group.add_argument('--more_epoch_train',
                       action='store_true', help='more_epoch_train')
    group.add_argument('--accumulation_steps', type=int,
                       default=1, help='accumulation_steps')
    group.add_argument('--digits', type=str,
                       default='float32', help='float32')
    group.add_argument('--clipnorm', type=float,default=1.0, help='clipnorm')
    group.add_argument('--loss_mag_weight', type=float,default=1.0, help='loss_mag_weight')
    group.add_argument('--loss_loc_weight', type=float,default=0.3, help='loss_loc_weight')
    return parser
def _structure_training_args(args):
    training_params=argparse.Namespace(
        batch_size= args.batch_size,
        continue_train = args.continue_train,
        do_train_shuffle = args.do_train_shuffle,
        seed=args.seed,
        pretrain_weight=args.pretrain_weight,
        epochs=args.epochs,
        use_amp=args.use_amp,
        job_mode=args.job_mode,
        save_warm_up=args.save_warm_up,
        save_every_epoch=args.save_every_epoch,
        epoch_save_list=args.epoch_save_list,
        more_epoch_train=args.more_epoch_train,
        accumulation_steps=args.accumulation_steps,
        digits=args.digits,
        clipnorm=args.clipnorm,
        loss_mag_weight=args.loss_mag_weight,
        loss_loc_weight=args.loss_loc_weight
    )
    return training_params

##############################################
############## Train Setting #################
##############################################
def _add_single_station_training_args(parser):
    group = parser.add_argument_group(title='single_station')
    group.add_argument('--single_station_train_config', type=str, default="config/train_stratagy/single_station.json",
                       help='the single_station config path')
    group.add_argument('--single_station_epochs', type=int, default=0,
                       help='single_station_epochs')
    return parser
def _structure_single_station_training_args(args):
    config = args.single_station_train_config
    if isinstance(config,str):
        with open(config, 'r') as f:
            config = json.load(f)
    else:
        assert isinstance(config,dict)
    if args.single_station_epochs:
        train = argparse.Namespace()
        for key, value in config['train'].items():
            setattr(train, key, value)
        train.epochs = args.single_station_epochs
        optimizer = argparse.Namespace()
        for key, value in config['optimizer'].items():
            setattr(optimizer, key, value)
        optimizer.epochs = train.epochs
        single_station_training_params=argparse.Namespace(
            train     = train,
            optimizer = optimizer,
        )
    else:
        single_station_training_params = None
    return single_station_training_params


##############################################
############## Valid Setting #################
##############################################
def _add_valid_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--valid_batch_size', type=int, default=2,help='valid batch size')
    group.add_argument('--valid_every_epoch', type=int, default=1,help='valid_every_epoch')
    group.add_argument('--evaluate_every_epoch', type=int,
                       default=10, help='evaluate_every_epoch')
    group.add_argument('--evaluate_branch', type=str,
                       default='TEST', help='evaluate_branch')
    group.add_argument('--sampling_rate', type=int, default=100,help='sampling_rate')
    return parser
def _structure_valid_args(args):
    validation_params=argparse.Namespace(
        valid_batch_size= args.valid_batch_size,
        valid_every_epoch=args.valid_every_epoch,
        evaluate_every_epoch=args.evaluate_every_epoch,
        sampling_rate=args.sampling_rate,
        evaluate_branch=args.evaluate_branch
    )
    return validation_params

##############################################
############### Loss Setting #################
##############################################
def _add_loss_args(parser):
    group = parser.add_argument_group(title='loss')
    parser.add_argument("--criterion_list", type=lambda s: tuple(map(str, s.split(","))), 
                        default="magnitude,location")
    return parser
def _structure_loss_args(args):
    criterion_params = argparse.Namespace(
        criterion_list= args.criterion,
    )
    return criterion_params

##############################################
############## Monitor Setting ###############
##############################################
def _add_monitor_args(parser):
    group = parser.add_argument_group(title='monitor')
    group.add_argument("--recorder_list", type=lambda s: list(map(str, s.split(","))),
                        default="tensorboard")
    group.add_argument('--disable_progress_bar',
                       action='store_true', help='whether continue train')
    group.add_argument("--log_trace_times", type=int, default=40)
    group.add_argument('--do_iter_log',
                       action='store_true', help='whether continue train')
    group.add_argument("--tracemodel", type=int, default=0)
    return parser
def _structure_monitor_args(args):
    monitor_params = argparse.Namespace(
        recorder_list= args.recorder_list,
        disable_progress_bar=args.disable_progress_bar,
        log_trace_times=args.log_trace_times,
        do_iter_log=args.do_iter_log,
        tracemodel=args.tracemodel
    )
    return monitor_params


##############################################
############# Optimizer Setting ##############
##############################################
def _add_optimizer_args(parser):
    group = parser.add_argument_group(title='learning rate')
    # Optimizer parameters # feed into timm
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=0, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument('--scheduler_inital_epochs', type=int, default=None)
    parser.add_argument('--scheduler_min_lr', type=float, default=None)
    parser.add_argument('--smooth_sigma_1', type=float, default=0.1)
    parser.add_argument('--smooth_sigma_2', type=float, default=2)
    parser.add_argument('--smooth_times', type=int, default=0)
    return parser
def _structure_optimizer_args(args):
    optimizer_params = argparse.Namespace(
        epochs=args.epochs,
        opt=args.opt,
        opt_eps=args.opt_eps,
        opt_betas=args.opt_betas,
        clip_grad=args.clip_grad,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr=args.lr,
        sched = args.sched,
        lr_noise = args.lr_noise,
        lr_noise_pct = args.lr_noise_pct,
        lr_noise_std = args.lr_noise_std,
        warmup_lr = args.warmup_lr,
        min_lr = args.min_lr,
        decay_epochs = args.decay_epochs,
        warmup_epochs = args.warmup_epochs,
        cooldown_epochs = args.cooldown_epochs,
        patience_epochs = args.patience_epochs,
        decay_rate = args.decay_rate,
        scheduler_inital_epochs = args.scheduler_inital_epochs,
        scheduler_min_lr = args.scheduler_min_lr,
        smooth_sigma_1 = args.smooth_sigma_1,
        smooth_sigma_2 = args.smooth_sigma_2,
        smooth_times = args.smooth_times,
    )
    return optimizer_params

##############################################
############## Dataset Setting ###############
##############################################
def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    return parser


##############################################
############## Dataset Setting ###############
##############################################
def _add_dataset_args(parser):
    group = parser.add_argument_group(title='data')
    group.add_argument("--overwrite_sampling_rate", type=int, default=None)
    group.add_argument("--data_configs", type=str, nargs='+')
    group.add_argument("--num_workers", type=int, default=32)
    group.add_argument("--windowlen", type=int, default=3000)
    group.add_argument("--dataset_alias", type=str, default=None)
    group.add_argument("--limit", type=int, default=None)
    group.add_argument("--use_single_processing_reading", action='store_true')
    group.add_argument("--target_type", type=str, default="all_in_one_source")
    
    return parser
def _structure_dataset_args(args):
    data_config_list = []
    for data_config in args.data_configs:
        if isinstance(data_config,str):
            with open(data_config, 'r') as f:
                config = json.load(f)
                assert 'data_path' in config, "the dataset configuration must contain the path of the data"
        else:
            config = data_config
        data_config_list.append(config)
    if len(data_config_list)==1:
        if "dataset_alias" in data_config_list[0]:
            args.dataset_alias = data_config_list[0]["dataset_alias" ]
            if "limit" in data_config_list[0]:
                args.limit         = data_config_list[0]["limit"]
    else:
        assert args.dataset_alias is not None, "multidataset requires dataset_alias"
    data_params = argparse.Namespace(
        overwrite_sampling_rate=args.overwrite_sampling_rate,
        data_configs = data_config_list,
        num_workers = args.num_workers,
        windowlen = args.windowlen,
        dataset_alias=args.dataset_alias,
        limit=args.limit,
        use_single_processing_reading=args.use_single_processing_reading,
        target_type=args.target_type
    )
    return data_params



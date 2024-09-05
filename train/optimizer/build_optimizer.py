import torch
from typing import Any, Callable, Sequence, Sized
from .optimizer_arguements import OptimizerConfig
from .sophia import SophiaG

def obtain_the_optimizer_grouped_parameters(parameters, no_decay_keywords,weight_decay):
    return [
        {
            'params': [p for n, p in parameters if (not any(nd in n for nd in no_decay_keywords) and p.requires_grad) ] ,
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if (any(nd in n for nd in no_decay_keywords) and p.requires_grad) ],
            'weight_decay': 0.0,
        },
    ]

def create_adamw_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-3,
    no_decay_keywords: Sequence[str] = ('bias', 'LayerNorm', 'layernorm'),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = obtain_the_optimizer_grouped_parameters(parameters, no_decay_keywords,weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def create_lion_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-3,
    no_decay_keywords: Sequence[str] = ('bias', 'LayerNorm', 'layernorm'),
):
    from lion_pytorch import Lion
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = obtain_the_optimizer_grouped_parameters(parameters, no_decay_keywords,weight_decay)
    optimizer = Lion(optimizer_grouped_parameters,lr=lr, use_triton=False)
    return optimizer

def create_sophia_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-3,
    no_decay_keywords: Sequence[str] = ('bias', 'LayerNorm', 'layernorm'),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = obtain_the_optimizer_grouped_parameters(parameters, no_decay_keywords,weight_decay)
    optimizer = SophiaG(optimizer_grouped_parameters, lr=lr)
    return optimizer


def get_optimizer(model,args: OptimizerConfig):
    if args.optimizer == 'adamw':
        optimizer = create_adamw_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'lion':
        optimizer = create_lion_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sophia':
        optimizer = create_sophia_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        
    else:
        raise NotImplementedError
    return optimizer
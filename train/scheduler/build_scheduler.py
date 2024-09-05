
from .scheduler_arguements import SchedulerConfig 
def get_scheduler(optimizer, train_dataloader, args:SchedulerConfig):
    if args.scheduler == 'none':
        lr_scheduler = None
    elif args.scheduler == 'TFcosine':
        from transformers import get_cosine_schedule_with_warmup  # type: ignore
        total_steps = len(train_dataloader) * args.epochs
        if num_warmup_steps < 1:
            num_warmup_steps = int(num_warmup_steps * total_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(num_warmup_steps),
            num_training_steps=total_steps,
        )
    elif args.scheduler == 'TimmCosine':
        from timm.scheduler import create_scheduler
        lr_scheduler, _ = create_scheduler(args.get_scheduler_config(), optimizer)

    return lr_scheduler

from dataclasses import dataclass
from simple_parsing import field
from train.optimizer.optimizer_arguements import OptimizerConfig
import argparse
@dataclass
class SchedulerConfig:
    scheduler: str = None
    lr: float = None # <--- assign it from optimizer
    epochs: int = None # <--- assign it from train controller
    
    def __post_init__(self):
        assert self.scheduler is not None, "It seem you are not assign one scheduler !!!! "
        
@dataclass
class NoSchedulerConfig(SchedulerConfig):
    scheduler: str = "none"

@dataclass
class TimmCosineConfig(SchedulerConfig):
    scheduler: str = "TimmCosine"
    warmup_epochs:int = field(default=10)
    sched_cycle_nums:int = field(default=1)
    scheduler_min_lr:float = field(default=1e-5)
    scheduler_warmup_lr:float = field(default=1e-5)
    def get_scheduler_config(self):
        assert self.lr is not None, "lr should be assigned from optimizer"
        assert self.epochs is not None, "epochs should be assigned from train controller"
        return argparse.Namespace(
            epochs = self.epochs//self.sched_cycle_nums,
            sched = 'cosine',
            min_lr = self.scheduler_min_lr,
            warmup_lr = min(self.scheduler_warmup_lr,self.lr),
            warmup_epochs = self.warmup_epochs,
            cooldown_epochs = 100,
            lr_cycle_limit = 100,
            lr_cycle_mul = 1,
        )
        

@dataclass
class TransformersCosineConfig(SchedulerConfig):
    scheduler: str = "TFCosine"
    num_warmup_steps: int = field(default=0)

    

        

    


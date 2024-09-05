from dataclasses import dataclass
from simple_parsing import field

@dataclass
class OptimizerConfig:
    optimizer: str = field(choices=["adamw", "sgd", "lion", "sophia"], default="adamw")
    lr: float = None # = field(default=1e-5) <-- must assign please,
    weight_decay : float = field(default=1e-2)
    

import os
import time
import numpy as np
import math
class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name = name
        self.count = 0
        self.mean = 0.0
        self.sum_squares = 0.0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed_time = time.time() - self.start_time
        self.update(elapsed_time)
        self.start_time = None

    def update(self, elapsed_time):
        self.count += 1
        delta = elapsed_time - self.mean
        self.mean += delta / self.count
        delta2 = elapsed_time - self.mean
        self.sum_squares += delta * delta2

    def mean_elapsed(self):
        return self.mean

    def std_elapsed(self):
        if self.count > 1:
            variance = self.sum_squares / (self.count - 1)
            return math.sqrt(variance)
        else:
            return 0.0

class Timers:
    """Group of timers."""

    def __init__(self, activate=False):
        self.timers = {}
        self.activate = activate
    def __call__(self, name):
        if not self.activate:return _DummyTimer()
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0):
        """Log a group of timers."""
        assert normalizer > 0.0
        print("Timer Results:")
        for name in names:
            mean_elapsed = self.timers[name].mean_elapsed() * 1000.0 / normalizer
            std_elapsed = self.timers[name].std_elapsed() * 1000.0 / normalizer
            space_num = " "*name.count('/')
            print(f"{space_num}{name}: {mean_elapsed:.2f}±{std_elapsed:.2f} ms")
class _DummyTimer:
    """A dummy timer that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
    
def think_up_a_unique_name_for_inference(args):#args:ProjectConfig
    assert args.task.train_or_infer in ['infer','infer_plot'], "train_or_infer must be infer or infer_plot, but got {}".format(args.task.train_or_infer)
    name = f"{args.task.sampling_strategy.valid_sampling_strategy.strategy_name}.w{args.task.sampling_strategy.valid_sampling_strategy.early_warning}.l{args.DataLoader.Dataset.max_length}"
    if hasattr(args.DataLoader.Dataset, 'component_intervel_length'):
        name = name + f".c{args.DataLoader.Dataset.component_intervel_length}"
    if hasattr(args.DataLoader.Dataset, 'padding_rule'):
        name = name + f".Pad_{args.DataLoader.Dataset.padding_rule}"
    return name
def get_local_rank(args=None):
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    return local_rank
def get_rank(args=None):
    local_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    return local_rank
def print0(*args,**kwargs):
    if get_rank()==0:print(*args,**kwargs)

def printg(*args,**kwargs):
    print(f"GPU:[{get_rank()}]",*args,**kwargs)


def smart_read_weight_path(path, device):
    import torch
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        weight = load_file(path)
        for k in weight.keys():
            weight[k] = weight[k].to(device)
        return weight
    return torch.load(path, map_location=device)

def smart_load_weight(model, weight, strict=True, shape_strict=True):
    model_dict = model.state_dict()
    has_not_used_key = False
    has_missing_key = False
    if not strict:
        for key in weight.keys():
            if key not in model_dict:
                print0(f"====> key: {key} are not used in this model, and we will skip")
                has_not_used_key = True
                continue
        if not has_not_used_key:print0("All keys in pretrained weight are used in this model")
        for key in model_dict.keys():
            if key not in weight:
                print0(f"====> key: {key} missing, please check. So far, we pass and random init it")
                has_missing_key = True
                continue
        if not has_missing_key:print0("All keys in this model are in pretrained weight")
            
    if shape_strict:
        model.load_state_dict(weight,strict=strict) 
    else:
        assert not strict, "shape_strict=False and strict=True is not allowed"
        for key in model_dict.keys():
            if key not in weight:
                print0(f"key: {key} missing, please check. So far, we pass and random init it")
                continue
            if model_dict[key].shape != weight[key].shape:
                print0(f"shape mismatching: {key} {model_dict[key].shape} {weight[key].shape}")
                continue
            model_dict[key] = weight[key]  
        
        
        model.load_state_dict(model_dict,strict=False)

def is_ancestor(ancestor, descendent):
    ancestor = os.path.abspath(ancestor)
    descendent = os.path.abspath(descendent)
    return descendent.startswith(ancestor)
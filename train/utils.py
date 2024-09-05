from __future__ import annotations
import torch
from accelerate import Accelerator
from tqdm import tqdm
import os
import logging


class DummyProgressBar:
    def __init__(*args,**kargs):
        pass 
    
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass


class DistributedTqdmProgressBar:
    def __init__(self, epochs: int, num_steps_per_epoch: int | None, bar_log_path=None, **kwargs) -> None:
        self.accelerator = Accelerator()
        self.epochs = epochs
        self.current_epoch = 1
        self.num_steps_per_epoch = num_steps_per_epoch
        self.tqdm_kwargs = kwargs
        self.bar_log  = self.create_logger('bar',console=False, offline_path=bar_log_path) if bar_log_path is not None and self.accelerator.is_main_process else None
    
    @staticmethod
    def create_logger(name,console=False,offline_path=None,console_level=logging.DEBUG,logformat='%(asctime)s %(message)s'):
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):logger.handlers.clear()# Important!!
        logger.setLevel(logging.DEBUG)
        
        if offline_path:
            info_dir,info_file = os.path.split(offline_path)
            os.makedirs(info_dir,exist_ok=True)
            handler = logging.FileHandler(offline_path)
            handler.setLevel(level = logging.DEBUG)
            handler.setFormatter(logging.Formatter(logformat))
            logger.addHandler(handler)

        if console:
            console = logging.StreamHandler()
            console.setLevel(console_level)
            console.setFormatter(logging.Formatter(logformat))
            logger.addHandler(console)
        return logger
    
    def on_epoch_start(self, current_epoch = None):
        if current_epoch is not None:self.current_epoch = current_epoch
        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(total=self.num_steps_per_epoch, **self.tqdm_kwargs)
        else:
            self.progress_bar = DummyProgressBar()

    def update(self, n: int = 1) -> None:
        self.progress_bar.update(n)

    def close(self) -> None:
        self.progress_bar.close()

    def on_epoch_end(self) -> None:
        self.current_epoch += 1
        if self.progress_bar is not None: self.progress_bar.close()

    def show_metrics(self, metrics: dict[str, float], iter_num=None) -> None:
        if iter_num is not None:
            description = f'Epoch {self.current_epoch}/{self.epochs}:[{iter_num}]'
        else:
            description = f'Epoch {self.current_epoch}/{self.epochs}'
        for name, score in metrics.items():
            if isinstance(score,float):
                description += f'|{name}:{score:.3f}'
            else:
                description += f'|{name}:{score}'
        self.progress_bar.set_description(description)
        if self.bar_log is not None:self.bar_log.info(description)

class LossTracker:
    def __init__(
        self,
        ndigits=4,
    ) -> None:
        self.ndigits = ndigits
        self._loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[float] = []

    def update(self, loss_tensor: torch.Tensor):
        
        loss = loss_tensor if isinstance(
            loss_tensor, float) else loss_tensor.item()
        self._loss = (self._loss * self.loss_count + loss) / \
            (self.loss_count + 1)
        self.loss_count += 1

    def reset(self):
        self._loss = 0
        self.loss_count = 0

    def on_epoch_end(self, reset: bool = True):
        self.history.append(self.loss)
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        return self._loss

def sendall2gpu(listinlist,device):
    if isinstance(listinlist,(list,tuple)):
        return [sendall2gpu(_list,device) for _list in listinlist]
    elif isinstance(listinlist, (dict)):
        return dict([(key,sendall2gpu(val,device)) for key,val in listinlist.items()])
    elif isinstance(listinlist, np.ndarray):
        return torch.from_numpy(listinlist).to(device=device, non_blocking=True)
    else:
        return listinlist.to(device=device, non_blocking=True)

class DataSimfetcher():
    def __init__(self, loader, device='auto'):
    
        if device == 'auto':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.loader = iter(loader)

    def next(self):
        try:

            self.batch = next(self.loader)
            self.batch = sendall2gpu(self.batch,self.device)
        except StopIteration:
            self.batch = None
        return self.batch

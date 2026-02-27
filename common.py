import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

random_seed = 42


@dataclass
class Config:
    subject_id: Optional[int] = None
    device: Optional[torch.device] = None
    patience: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    kfold_n_splits: Optional[int] = None
    lr: Optional[float] = None
    adamw_eps: Optional[float] = None
    weight_decay: Optional[float] = None
    fmin: Optional[int] = None
    fmax: Optional[int] = None
    remove_bad_trial: Optional[bool] = None
    scheduler: Optional[str] = None


def set_seed(seed=random_seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

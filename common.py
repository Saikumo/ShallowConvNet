import random
from dataclasses import dataclass

import numpy as np
import torch

random_seed = 42


@dataclass
class Config:
    subject_id: int
    device: torch.device
    patience: int
    epochs: int
    batch_size: int
    kfold_n_splits: int
    lr: float
    adamw_eps: float
    weight_decay: float


def set_seed(seed=random_seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

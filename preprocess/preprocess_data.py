from moabb.datasets import BNCI2014_001
import torch
import mne
from moabb.paradigms import MotorImagery
from sklearn.model_selection import KFold

import common


def preprocess_bnci2014_001(subject_id):
    X_train, y_train = load_bnci2014_001_data_from_moabb(subject_id=subject_id, train=True)
    X_test, y_test = load_bnci2014_001_data_from_moabb(subject_id=subject_id, train=False)

    mean = X_train.mean(dim=(0, 2), keepdim=True)
    std = X_train.std(dim=(0, 2), keepdim=True)

    X_train = (X_train - mean) / (std + 1e-9)
    X_test = (X_test - mean) / (std + 1e-9)

    return X_train, y_train, X_test, y_test


def preprocess_kfold_bnci2014_001(subject_id, n_splits=5, random_state=common.random_seed):
    X, y = load_bnci2014_001_data_from_moabb(subject_id, train=True)

    folds = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        mean = X_train.mean(dim=(0, 2), keepdim=True)
        std = X_train.std(dim=(0, 2), keepdim=True)

        X_train = (X_train - mean) / (std + 1e-9)
        X_val = (X_val - mean) / (std + 1e-9)

        folds.append({
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        })

    return folds


def get_bnci2014_001_event_id():
    event_id = BNCI2014_001().event_id
    # 将 event_id 的值映射到 0,1,2,3
    min_val = min(event_id.values())
    event_id = {k: v - min_val for k, v in event_id.items()}
    return event_id


def load_bnci2014_001_data_from_moabb(subject_id, train, fmin=0, fmax=38, tmin=-0.5, tmax=4):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    X_all, labels_all, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    session = '0train' if train else '1test'
    X = X_all[metadata['session'] == session]
    X = X * 1e6 # unit V to uV
    labels = labels_all[metadata['session'] == session]
    y = [get_bnci2014_001_event_id()[label] for label in labels]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

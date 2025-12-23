from moabb.datasets import BNCI2014_001
import torch
import mne
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
    run = BNCI2014_001().get_data(subjects=[1])[1]['0train']['0']
    events, event_id = mne.events_from_annotations(run)
    # 将 event_id 的值映射到 0,1,2,3
    min_val = min(event_id.values())
    event_id = {k: v - min_val for k, v in event_id.items()}
    return event_id


def load_bnci2014_001_data_from_moabb(subject_id, train):
    dataset = BNCI2014_001()
    subject_data = dataset.get_data(subjects=[subject_id])
    session = '0train' if train else '1test'

    X_list, y_list = [], []

    for run_id in subject_data[subject_id][session].keys():
        run = subject_data[subject_id][session][run_id]
        X_run, y_run = extract_raw(run)
        X_list.append(X_run)
        y_list.append(y_run)

    X_all = torch.cat(X_list, dim=0)  # (N, C, T)
    y_all = torch.cat(y_list, dim=0)  # (N,)
    return X_all, y_all


def extract_raw(
        run,
        tmin=0.0,
        tmax=4.0,
        l_freq=4.0,
        h_freq=38.0,
):
    """
    run: mne.io.BaseRaw (Raw / RawArray / RawBrainVision)
    return:
        X: torch.FloatTensor (n_trials, n_channels, n_times)
        y: torch.LongTensor  (n_trials,)
    """

    # 1. 从 annotations 中提取事件
    events, event_id = mne.events_from_annotations(run)

    # 3. 构造 Epochs（标准做法）
    epochs = mne.Epochs(
        run,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # 3. 带通滤波
    epochs.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        verbose=False,
    )

    # 4. 取数据
    X = epochs.get_data()  # (n_trials, n_channels, n_times)
    y = epochs.events[:, -1]  # label id
    y = y - y.min()  # 映射到 0,1,2,3

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

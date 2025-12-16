import torch
from torch.utils.data import Dataset, DataLoader
from moabb.datasets import BNCI2014_001
import numpy as np


class BNCI2014_001_Dataset(Dataset):
    def __init__(self, train,mean,std):
        # 选择数据集和任务
        dataset = BNCI2014_001()
        subject1 = dataset.get_data(subjects=[1])
        # run0 = subject1[1]['0train' if train else '1test']['0']
        run0 = subject1[1]['0train']['1' if train else '0']
        if train:
            X1,y1 = convert(subject1[1]['0train']['0'])
            X2,y2 = convert(subject1[1]['0train']['1'])
            X3,y3 = convert(subject1[1]['0train']['2'])
            X4,y4 = convert(subject1[1]['0train']['3'])
            X5,y5 = convert(subject1[1]['0train']['4'])
            X6,y6 = convert(subject1[1]['0train']['5'])
            self.X = torch.cat([X1,X2,X3,X4,X5,X6],dim=0)
            self.y = torch.cat([y1,y2,y3,y4,y5,y6],dim=0)
        else:
            X1, y1 = convert(subject1[1]['1test']['0'])
            X2, y2 = convert(subject1[1]['1test']['1'])
            X3, y3 = convert(subject1[1]['1test']['2'])
            X4, y4 = convert(subject1[1]['1test']['3'])
            X5, y5 = convert(subject1[1]['1test']['4'])
            X6, y6 = convert(subject1[1]['1test']['5'])
            self.X = torch.cat([X1, X2, X3, X4, X5, X6], dim=0)
            self.y = torch.cat([y1, y2, y3, y4, y5, y6], dim=0)

        if mean is None or std is None:
            # 如果没有提供，则用当前数据集自己计算
            self.mean = self.X.mean(dim=(0, 2), keepdim=True)  # 每个通道的均值
            self.std = self.X.std(dim=(0, 2), keepdim=True)
        else:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X - self.mean) / (self.std + 1e-10)
        y = self.y
        return x[idx], y[idx]

    def get_shape(self):
        return self.X.shape

    def get_mean_std(self):
        return (self.mean, self.std)

def convert(run0):
    sfreq = run0.info['sfreq']  # 采样率 Hz
    X_list = []
    y_list = []
    # 假设你的所有可能标签如下：
    label_names = ["left_hand", "right_hand", "feet", "tongue"]
    # 建 label → int 的映射字典
    label2int = {label: idx for idx, label in enumerate(label_names)}
    # 比如: {"left_hand": 0, "right_hand": 1, "foot": 2, "tongue": 3}

    for annot in run0.annotations:
        onset_sample = int(annot['onset'] * sfreq)  # 起始点
        duration_sample = int(annot['duration'] * sfreq)  # 持续点数
        trial_data = run0.get_data()[:, onset_sample: onset_sample + duration_sample]
        X_list.append(trial_data)
        label_str = annot['description']
        y_list.append(label2int[label_str])

    # 转成 numpy array
    X = np.array(X_list)  # shape: (n_trials, n_channels, n_times)
    y = np.array(y_list)
    return torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long)


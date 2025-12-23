import numpy as np
import torch

from preprocess_data import preprocess_bnci2014_001
from eeg_dataset import EEGDataset
from torch.utils.data import DataLoader
import shallow_convnet
from train_one_epoch import train_one_epoch, eval_one_epoch


def train(device, epochs=34, batch_size=64, lr=1e-3):
    losses = []
    accs = []

    for i in range(9):
        X_train, y_train, X_test, y_test = preprocess_bnci2014_001(i + 1)
        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=(device.type == "cuda"))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

        model = shallow_convnet.ShallowConvNet(X_train.shape)
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

            print(f"Subject {i + 1}, Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | ")

        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
        losses.append(test_loss)
        accs.append(test_acc)
        print(f"Subject {i + 1}, Epoch {epoch + 1}/{epochs} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.3f}")

    # 计算均值和标准差（loss 和 accuracy）
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    # 打印结果
    print(f"Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

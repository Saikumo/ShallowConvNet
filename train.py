import numpy as np
import torch

import common
from preprocess.preprocess_data import preprocess_bnci2014_001
from eeg_dataset import EEGDataset
from torch.utils.data import DataLoader

from shallow_convnet_speedup import ShallowConvNetSpeedup
from train_one_epoch import train_one_epoch, eval_one_epoch


def train(device, batch_size=64, patience=20, epochs=500):
    losses = []
    accs = []
    kappas = []
    best_epochs = []
    best_losses = []
    best_loss_accs = []
    best_losses_kappas = []

    for i in range(9):
        X_train, y_train, X_test, y_test = preprocess_bnci2014_001(i + 1)

        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=common.random_seed,
            stratify=y_train  # 关键！保持类别比例
        )

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=(device.type == "cuda"), num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=(device.type == "cuda"), num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"),
                                 num_workers=4, persistent_workers=True)

        model = ShallowConvNetSpeedup()
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1 * 0.01, eps=1e-8, weight_decay=0)

        best_loss = float("inf")
        best_epoch = 0
        best_loss_acc = float("inf")
        counter = 0
        best_loss_kappa = float("inf")
        best_test_loss = float("inf")
        best_test_acc = float("inf")
        best_test_kappa = float("inf")

        for epoch in range(epochs):
            train_loss, train_acc, train_kappa = train_one_epoch(model, train_loader, optimizer, None, criterion,
                                                                 device)
            val_loss, val_acc, val_kappa = eval_one_epoch(model, val_loader, criterion, device)

            print(f"Fold {i + 1}, Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Train Kappa: {train_kappa:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Val Kappa: {val_kappa:.4f}")

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_loss_acc = val_acc
                counter = 0
                best_loss_kappa = val_kappa
                test_loss, test_acc, test_kappa = eval_one_epoch(model, test_loader, criterion, device)
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_test_kappa = test_kappa

                print(f"Subject{i + 1}, Epoch {epoch + 1}/{epochs} | "
                      f"Test Loss: {test_loss:.4f} Acc: {test_acc:.3f} Test Kappa: {test_kappa:.3f}")
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping")
                best_epochs.append(best_epoch)
                best_losses.append(best_loss)
                best_loss_accs.append(best_loss_acc)
                best_losses_kappas.append(best_loss_kappa)

                losses.append(best_test_loss)
                accs.append(best_test_acc)
                kappas.append(best_test_kappa)
                break




    # 计算均值和标准差（loss 和 accuracy）
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)

    # 打印结果
    print(f"Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")

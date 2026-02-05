from eeg_dataset import EEGDataset
from preprocess.preprocess_data import *
from shallow_convnet_speedup import ShallowConvNetSpeedup
from train_one_epoch import *
import numpy as np
from torch.utils.data import DataLoader


def train_kfold(device, subjectId=1, patience=20, epochs=100, batch_size=64):
    folds = preprocess_kfold_bnci2014_001(subject_id=subjectId, n_splits=5)

    print(f"device {device},subject {subjectId}")

    best_epochs = []
    best_losses = []
    best_loss_accs = []
    best_losses_kappas = []

    for i, fold in enumerate(folds):
        X_train = fold['X_train']
        y_train = fold['y_train']
        X_val = fold['X_val']
        y_val = fold['y_val']
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=(device.type == 'cuda'), num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=(device.type == 'cuda'), num_workers=4, persistent_workers=True)

        model = ShallowConvNetSpeedup()
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0625 * 0.01, eps=1e-8)

        best_loss = float("inf")
        best_epoch = 0
        best_loss_acc = float("inf")
        counter = 0
        best_loss_kappa = float("inf")

        for epoch in range(epochs):
            train_loss, train_acc, train_kappa = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_kappa = eval_one_epoch(model, val_loader, criterion, device)

            # print(f"Fold {i + 1}, Epoch {epoch + 1}/{epochs} | "
            #       f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            #       f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_loss_acc = val_acc
                counter = 0
                best_loss_kappa = val_kappa
                print(f"Fold {i + 1}, Epoch {epoch + 1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Train Kappa: {train_kappa:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Val Kappa: {val_kappa:.4f}")
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping")
                best_epochs.append(best_epoch)
                best_losses.append(best_loss)
                best_loss_accs.append(best_loss_acc)
                best_losses_kappas.append(best_loss_kappa)
                break

    # 取 epoch 的中位数作为最终训练 epoch
    median_epoch = int(np.median(best_epochs))
    # 计算均值和标准差（loss 和 accuracy）
    mean_loss = np.mean(best_losses)
    std_loss = np.std(best_losses)
    mean_acc = np.mean(best_loss_accs)
    std_acc = np.std(best_loss_accs)
    mean_kappa = np.mean(best_losses_kappas)
    std_kappa = np.std(best_losses_kappas)
    # 打印结果
    print(f"Median Epoch for subject{subjectId}: {median_epoch}")
    print(f"Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Accuracy at Best Loss: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Kappa at Best Loss: {mean_kappa:.4f} ± {std_kappa:.4f}")
    return median_epoch


def train_all_kfold():
    best_epochs = []

    for i in range(9):
        epoch = train_kfold(subjectId=i + 1, device=torch.device("cuda"))
        best_epochs.append(epoch)

    median_epoch = int(np.median(best_epochs))
    print(f"Median Epoch for final training: {median_epoch}")

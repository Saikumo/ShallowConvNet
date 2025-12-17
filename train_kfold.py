from eeg_dataset import EEGDataset
from preprocess_data import *
from train import *
import numpy as np
from torch.utils.data import DataLoader
import shallow_convnet


def train_kfold(patience=20, epochs=100, batch_size=64):
    folds = preprocess_kfold_bnci2014_001(subject_id=1, n_splits=5, random_state=42)

    best_epochs = []
    best_losses = []
    best_loss_accs = []

    for i, fold in enumerate(folds):
        X_train = fold['X_train']
        y_train = fold['y_train']
        X_val = fold['X_val']
        y_val = fold['y_val']
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = shallow_convnet.ShallowConvNet(X_train.shape)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_loss = float("inf")
        best_epoch = 0
        best_loss_acc = float("inf")
        counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

            print(f"Fold {i + 1}, Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_loss_acc = val_acc
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping")
                best_epochs.append(best_epoch)
                best_losses.append(best_loss)
                best_loss_accs.append(best_loss_acc)
                break

    # 取 epoch 的中位数作为最终训练 epoch
    median_epoch = int(np.median(best_epochs))
    # 计算均值和标准差（loss 和 accuracy）
    mean_loss = np.mean(best_losses)
    std_loss = np.std(best_losses)
    mean_acc = np.mean(best_loss_accs)
    std_acc = np.std(best_loss_accs)
    # 打印结果
    print(f"Median Epoch for final training: {median_epoch}")
    print(f"Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Accuracy at Best Loss: {mean_acc:.4f} ± {std_acc:.4f}")

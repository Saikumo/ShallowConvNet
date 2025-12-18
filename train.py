import numpy as np
import torch
from preprocess_data import preprocess_bnci2014_001
from eeg_dataset import EEGDataset
from torch.utils.data import DataLoader
import shallow_convnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)  # 输出 shape = (batch, 4)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            total_loss += criterion(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total


def train(epochs=10, batch_size=64, lr=1e-3):
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
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)

            print(f"Subject {i + 1}, Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | ")

        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
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

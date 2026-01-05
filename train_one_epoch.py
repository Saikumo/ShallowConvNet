import torch
from sklearn.metrics import cohen_kappa_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)  # 输出 shape = (batch, 4)
        y_crop = y.repeat_interleave(619)

        loss = criterion(logits, y_crop)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        B = y.size(0)
        logits_trial = logits.view(B, 619, 4).mean(dim=1)
        pred = logits_trial.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_preds.append(pred.detach().cpu())
        all_labels.append(y.detach().cpu())

    all_preds = torch.cat(all_preds).flatten().numpy()
    all_labels = torch.cat(all_labels).flatten().numpy()

    return total_loss / len(loader), correct / total, cohen_kappa_score(all_preds, all_labels)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            y_crop = y.repeat_interleave(619)

            total_loss += criterion(logits, y_crop).item()

            B = y.size(0)
            logits_trial = logits.view(B, 619, 4).mean(dim=1)
            pred = logits_trial.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            all_preds.append(pred.detach().cpu())
            all_labels.append(y.detach().cpu())

    all_preds = torch.cat(all_preds).flatten().numpy()
    all_labels = torch.cat(all_labels).flatten().numpy()

    return total_loss / len(loader), correct / total, cohen_kappa_score(all_preds, all_labels)

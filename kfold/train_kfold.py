from common import Config
from eeg_dataset import EEGDataset
from preprocess.preprocess_data import *
from shallow_convnet_speedup import ShallowConvNetSpeedup
from train_one_epoch import *
import numpy as np
from torch.utils.data import DataLoader
import wandb
import os
from dataclasses import dataclass
from datetime import datetime
from dataclasses import asdict


def train_kfold(config):
    folds = preprocess_kfold_bnci2014_001(subject_id=config.subject_id, n_splits=config.kfold_n_splits)

    best_losses = []
    best_loss_accs = []
    best_losses_kappas = []

    for i, fold in enumerate(folds):
        run = wandb.init(
            entity="saikumo11-saikumo-s",
            project="ShallowConvNet-KFold",
            config={**asdict(config), "fold": i + 1},
        )

        X_train, y_train, X_val, y_val = (fold[k] for k in ['X_train', 'y_train', 'X_val', 'y_val'])

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  pin_memory=(config.device.type == 'cuda'), num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                pin_memory=(config.device.type == 'cuda'), num_workers=4, persistent_workers=True)

        model = ShallowConvNetSpeedup()
        model.to(config.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=config.adamw_eps,
                                      weight_decay=config.weight_decay)

        best_loss = float("inf")
        best_loss_acc = float("inf")
        best_loss_kappa = float("inf")
        counter = 0

        for epoch in range(config.epochs):
            train_loss, train_acc, train_kappa = train_one_epoch(model, train_loader, optimizer, None, criterion,
                                                                 config.device)
            val_loss, val_acc, val_kappa = eval_one_epoch(model, val_loader, criterion, config.device)

            run.log(
                {"train_loss_": train_loss, "train_acc_": train_acc, "train_kappa_": train_kappa, "val_loss_": val_loss,
                 "val_acc_": val_acc, "val_kappa_": val_kappa})

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_loss_acc = val_acc
                best_loss_kappa = val_kappa
                counter = 0
            else:
                counter += 1

            if counter >= config.patience:
                print("Early stopping")
                best_losses.append(best_loss)
                best_loss_accs.append(best_loss_acc)
                best_losses_kappas.append(best_loss_kappa)
                break
        run.finish()

    # 计算均值和标准差（loss 和 accuracy）
    mean_loss = np.mean(best_losses)
    std_loss = np.std(best_losses)
    mean_acc = np.mean(best_loss_accs)
    std_acc = np.std(best_loss_accs)
    mean_kappa = np.mean(best_losses_kappas)
    std_kappa = np.std(best_losses_kappas)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_run = wandb.init(
        entity="saikumo11-saikumo-s",
        project="ShallowConvNet-KFold",
        name=f"subject{config.subject_id}_summary_{timestamp}",
        config=config,
    )
    summary_run.log({"mean_loss": mean_loss, "std_loss": std_loss, "mean_acc": mean_acc, "std_acc": std_acc,
                     "mean_kappa": mean_kappa, "std_kappa": std_kappa})
    summary_run.finish()


def train_all_kfold():
    wandb.login(key=os.environ["WANDB_API_KEY"])

    for i in range(9):
        config = Config(
            subject_id=i + 1,
            device=torch.device("cuda"),
            patience=20,
            epochs=500,
            batch_size=64,
            kfold_n_splits=5,
            lr=0.0625 * 0.01,
            adamw_eps=1e-8,
            weight_decay=0,
        )
        train_kfold(config)

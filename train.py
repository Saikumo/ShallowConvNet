from datetime import datetime

import numpy as np
import torch

import common
from preprocess.preprocess_data import preprocess_train_bnci2014_001
from eeg_dataset import EEGDataset
from torch.utils.data import DataLoader

from shallow_convnet_speedup import ShallowConvNetSpeedup
from train_one_epoch import train_one_epoch, eval_one_epoch
import wandb


def train(device):
    losses = []
    accs = []
    kappas = []

    config = common.Config(
        device=torch.device("cuda"),
        patience=50,
        epochs=500,
        batch_size=64,
        lr=0.0625 * 0.01,
        adamw_eps=1e-8,
        weight_decay=0,
        fmin=0,
        fmax=38,
        remove_bad_trial=False
    )

    for i in range(9):
        # different subject different config
        config.subject_id = i + 1
        config.remove_bad_trial = False
        config.fmax = 38
        if config.subject_id == 2:
            config.fmax = 100
        elif config.subject_id == 5:
            config.remove_bad_trial = True

        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_train_bnci2014_001(i + 1, config.fmin, config.fmax,
                                                                                       config.remove_bad_trial)

        train_dataset, val_dataset, test_dataset = EEGDataset(X_train, y_train), EEGDataset(X_val, y_val), EEGDataset(
            X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  pin_memory=(device.type == "cuda"), num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                pin_memory=(device.type == "cuda"), num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                 pin_memory=(device.type == "cuda"),
                                 num_workers=4, persistent_workers=True)

        model = ShallowConvNetSpeedup()
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=config.adamw_eps,
                                      weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,  # lr *= 0.5
            patience=5,  # 5 epoch 不下降就降
            min_lr=1e-5,
            verbose=True
        )

        best_loss = float("inf")
        counter = 0
        best_test_loss = float("inf")
        best_test_acc = float("inf")
        best_test_kappa = float("inf")

        run = wandb.init(
            entity="saikumo11-saikumo-s",
            project="ShallowConvNet",
            config=config
        )

        for epoch in range(config.epochs):
            train_loss, train_acc, train_kappa = train_one_epoch(model, train_loader, optimizer, None, criterion,
                                                                 device)
            val_loss, val_acc, val_kappa, _, _ = eval_one_epoch(model, val_loader, criterion, device)
            test_loss, test_acc, test_kappa, test_preds, test_labels = eval_one_epoch(model, test_loader, criterion,
                                                                                      device)
            scheduler.step(val_loss)

            run.log(
                {"train_loss_": train_loss, "train_acc_": train_acc, "train_kappa_": train_kappa, "val_loss_": val_loss,
                 "val_acc_": val_acc, "val_kappa_": val_kappa, "test_loss": test_loss, "test_acc": test_acc,
                 "test_kappa": test_kappa}, step=epoch + 1)

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                counter = 0

                best_test_loss = test_loss
                best_test_acc = test_acc
                best_test_kappa = test_kappa
                run.log(
                    {"confusion_matrix": wandb.plot.confusion_matrix(
                        preds=test_preds,
                        y_true=test_labels
                    )}, step=epoch + 1)

            else:
                counter += 1

            if counter >= config.patience:
                print("Early stopping")
                losses.append(best_test_loss)
                accs.append(best_test_acc)
                kappas.append(best_test_kappa)
                run.log(
                    {"best_test_loss": best_test_loss, "best_test_acc": best_test_acc,
                     "best_test_kappa": best_test_kappa})
                break
        run.finish()

    # 计算均值和标准差（loss 和 accuracy）
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)

    config.subject_id = None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_run = wandb.init(
        entity="saikumo11-saikumo-s",
        project="ShallowConvNet",
        name=f"train_summary_{timestamp}",
        config=config,
    )
    summary_run.log({"mean_loss": mean_loss, "std_loss": std_loss, "mean_acc": mean_acc, "std_acc": std_acc,
                     "mean_kappa": mean_kappa, "std_kappa": std_kappa})
    summary_run.finish()

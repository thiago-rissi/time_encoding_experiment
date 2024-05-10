from dataset.datasets import TorchDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from typing import Any
import torch.nn as nn
import pathlib
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import numpy as np
import pickle


def save_model(model, outputfile: pathlib.Path):
    torch.save(model.state_dict(), outputfile)


def index_agreement_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
    index of agreement
    Willmott (1981, 1982)

    Args:
        s: simulated
        o: observed

    Returns:
        ia: index of agreement
    """
    o_bar = torch.mean(o, dim=0)
    ia = 1 - (torch.sum((o - s) ** 2, dim=0)) / (
        torch.sum(
            (torch.abs(s - o_bar) + torch.abs(o - o_bar)) ** 2,
            dim=0,
        )
    )

    return ia.mean()


class TorchTrainer:
    def __init__(
        self,
        loss: str,
        model: nn.Module,
        **kwargs,
    ) -> None:
        self.model = model
        self.loss_func = getattr(nn, loss)()

    def step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        timestamps: torch.Tensor,
        optimizer: Optimizer,
    ):
        optimizer.zero_grad()
        y_hat = self.model(X, timestamps)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        optimizer.step()

        return loss

    def train_epoch(
        self,
        train_dl: DataLoader,
        validation_dl: DataLoader,
        optimizer: Adam,
        device: torch.device,
    ):
        losses = []
        self.model.train()
        for i, (X, y, timestamps) in enumerate((pbar := tqdm(train_dl))):
            X.to(device)
            y.to(device)
            loss = self.step(
                X=X,
                y=y,
                timestamps=timestamps,
                optimizer=optimizer,
            )
            pbar.set_description(f"{i+1}/{len(train_dl)}-{loss.item():.4f}")

        losses = []

        self.model.eval()
        with torch.no_grad():
            for i, (X, y, timestamps) in enumerate((pbar := tqdm(validation_dl))):
                X.to(device)
                y.to(device)
                y_hat = self.model(X, timestamps)
                loss = self.loss_func(y_hat, y)
                losses.append(loss.item())
        return np.mean(losses)

    def train(
        self,
        dataset: TorchDataset,
        n_epochs: int,
        batch_size: int,
        early_stop: bool,
        patience: int,
        lr: float,
        save_path: str,
        snapshot_interval: int,
        device: torch.device,
        num_workers: int = 0,
        **kwargs,
    ) -> None:

        device_ = torch.device(device)
        self.model.to(device_)

        train_dataset, validation_dataset = dataset.split_dataset(0.9)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

        optimizer = Adam(
            self.model.parameters(),
            lr=float(lr),
        )

        best_loss = torch.inf
        for epoch in range(n_epochs):
            loss = self.train_epoch(
                train_dl=train_dataloader,
                validation_dl=validation_dataloader,
                optimizer=optimizer,
                device=device_,
            )

            if epoch % snapshot_interval == 0:
                save_model(self.model, pathlib.Path(save_path) / f"model_{epoch}.pkl")

            if early_stop:
                if loss < best_loss:
                    save_model(
                        self.model, pathlib.Path(save_path) / f"model_{epoch}_best.pkl"
                    )
                    best_loss = loss
                else:
                    patience -= 1

                if patience == 0:
                    break

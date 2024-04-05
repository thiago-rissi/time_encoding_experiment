from dataset.datasets import DeepDataset
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


class DeepTrainer:
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
            losses.append(loss.item())
            pbar.set_description(f"{i+1}/{len(train_dl)}-{loss.item():.4f}")
        return np.mean(losses)

    def train(
        self,
        dataset: DeepDataset,
        n_epochs: int,
        batch_size: int,
        early_stop: bool,
        tol: float,
        lr: float,
        save_path: str,
        snapshot_interval: int,
        device: torch.device,
        num_workers: int = 0,
        **kwargs,
    ) -> None:

        device_ = torch.device(device)
        self.model.to(device_)

        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        optimizer = Adam(
            self.model.parameters(),
            lr=float(lr),
        )

        for epoch in range(n_epochs):
            loss = self.train_epoch(
                train_dl=train_dataloader,
                optimizer=optimizer,
                device=device_,
            )

            if epoch % snapshot_interval == 0:
                save_model(self.model, pathlib.Path(save_path) / f"model_{epoch}.pkl")

            if early_stop:
                if abs(loss) < float(tol):
                    save_model(
                        self.model, pathlib.Path(save_path) / f"model_{epoch}.pkl"
                    )
                    break

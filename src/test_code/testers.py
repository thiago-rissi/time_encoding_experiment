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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import polars as pl


class TorchTester:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        loss: str,
        **kwargs,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.loss_func = getattr(nn, loss)()

    def predict(
        self, X: torch.Tensor, y: torch.Tensor, timestamps: torch.Tensor
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        self.model.eval()
        y_hat = self.model(X, timestamps)
        loss = self.loss_func(y_hat, y)

        return y_hat, loss

    def test(
        self,
        dataset: TorchDataset,
        batch_size: int,
        device: torch.device,
        save_path: str,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        device_ = torch.device(device)
        self.model.to(device_)

        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )
        ys = []
        ys_hat = []
        for i, (X, y, timestamps) in enumerate((pbar := tqdm(test_dataloader))):
            with torch.no_grad():
                y_hat, loss = self.predict(X, y, timestamps)
                y_hat = np.argmax(y_hat.cpu(), axis=1)
                y = y.cpu()
                ys.append(y)
                ys_hat.append(y_hat)

        y = np.concatenate(ys)
        y_hat = np.concatenate(ys_hat)
        print(accuracy_score(y, y_hat))
        results = pl.DataFrame({"y": y, "y_hat": y_hat})
        results.write_parquet(pathlib.Path(save_path))

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
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)
        loss = self.loss_func(y_hat, y)

        return y_hat, loss

    def test(
        self,
        dataset: TorchDataset,
        device: torch.device,
        save_path: str,
        inf_sample_size: int,
        num_workers: int = 0,
        **kwargs,
    ) -> float:
        device_ = torch.device(device)
        self.model.to(device_)

        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )
        all_infs = []
        for inf_sample in range(inf_sample_size):
            ys = []
            ys_hat = []
            for i, (X, y, timestamps) in enumerate((pbar := tqdm(test_dataloader))):
                with torch.no_grad():
                    y_hat, loss = self.predict(X, y, timestamps)
                    y = y.cpu()
                    ys.append(y)
                    ys_hat.append(y_hat)

            y = torch.cat(ys)
            y_hat = torch.stack(ys_hat)
            all_infs.append(y_hat)
        y_hat = torch.mean(torch.stack(all_infs), dim=0)

        y_hat = torch.argmax(y_hat.cpu(), dim=-1)

        acc = accuracy_score(y, y_hat)
        print(acc)
        results = pl.DataFrame({"y": y.numpy(), "y_hat": (y_hat.squeeze()).numpy()})

        s = pathlib.Path(save_path)
        s.parent.mkdir(exist_ok=True, parents=True)
        results.write_parquet(s)
        return acc

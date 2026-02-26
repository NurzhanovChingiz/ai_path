"""Training functions for PyTorch models."""

from typing import cast
from collections.abc import Sized

import torch
from torch import nn
from torch.utils.data import DataLoader


def train(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> None:
    """Train the model.

    Args:
        model: The model to train.
        dataloader: The dataloader to train the model on.
        loss_fn: The loss function to use.
        optimizer: The optimizer to use.
        device: The device to train the model on.

    Returns:
        None
    """
    model.train()
    batch_idx: int = 0
    size: int = len(cast(Sized, dataloader.dataset))
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

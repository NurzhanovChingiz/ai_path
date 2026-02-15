# Train function
import torch
from torch import nn
from torch.utils.data import DataLoader

def train(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """
    Train the model.

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
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
    assert dataloader.dataset is not None
    size = len(dataloader.dataset)  # type: ignore[arg-type]

    model.train()  
    for b, (X, y) in enumerate(dataloader):  
        X, y = X.to(device), y.to(
            device)  
        optimizer.zero_grad()  
        pred = model(X)  

        loss = loss_fn(pred, y)  

        loss.backward()  
        optimizer.step()  
        if (b + 1) % 100 == 0:
            loss, current = loss.item(), b * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

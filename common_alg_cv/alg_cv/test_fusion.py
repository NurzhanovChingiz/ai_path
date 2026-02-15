import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from .fusion import fuse_model_inplace

def test(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device, mode: str = "Test") -> None:
    """
    Test the model.

    Args:
        model: The model to test.
        dataloader: The dataloader to test the model on.
        loss_fn: The loss function to use.
        device: The device to test the model on.
        mode: The mode to test the model on.

    Returns:
        None
    """
    tmp = copy.deepcopy(model).to(device)
    tmp.eval()

    fuse_model_inplace(tmp)
    test_loss: float = 0.0
    correct: float = 0.0
    total: int = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = tmp(X)
            test_loss += loss_fn(pred, y).item()
            total += y.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss = test_loss / max(batch_idx, 1)

    correct = 100.*correct/total
    print(f"{mode} Error:\n Accuracy: {(correct):>.1f}%, Avg loss: {(loss):>.8f}\n")


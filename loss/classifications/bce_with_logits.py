# binary cross entropy with logits
import numpy as np 
import torch
from torch.nn import functional as F
from torch import nn

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1 / (1 + np.exp(-x))
    return result

def bce_with_logits_np(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Numpy implementation of binary cross entropy with logits.
    y: class indices of shape (N,)
    y_pred: raw logits of shape (N,)
    """
    sigma = sigmoid_np(y_pred)
    bce: np.ndarray = -1 * (y * np.log(sigma) + (1 - y) * np.log(1 - sigma))
    result: float = bce.mean()
    return result

def bce_with_logits_np_stable(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Numpy implementation of binary cross entropy with logits.
    y: class indices of shape (N,)
    y_pred: raw logits of shape (N,)
    """
    # max(x, 0)
    max_val = np.maximum(y_pred, 0)

    # log(1 + exp(-|x|))
    log_term = np.log1p(np.exp(-np.abs(y_pred)))

    loss = max_val - y_pred * y + log_term
    result: float = loss.mean()
    return result

def sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    result: torch.Tensor = 1 / (1 + torch.exp(-x))
    return result

def bce_with_logits_torch(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of binary cross entropy with logits.
    y: class indices of shape (N,)
    y_pred: raw logits of shape (N,)
    """
    sigma = sigmoid_torch(y_pred)
    bce = -1 * (y * torch.log(sigma) + (1 - y) * torch.log(1 - sigma))
    result: torch.Tensor = bce.mean()
    return result

def bce_with_logits_torch_stable(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of binary cross entropy with logits.
    y: class indices of shape (N,)
    y_pred: raw logits of shape (N,)
    """
    max_val = torch.maximum(y_pred, torch.zeros_like(y_pred))
    log_term = torch.log1p(torch.exp(-torch.abs(y_pred)))
    loss = max_val - y_pred * y + log_term
    result: torch.Tensor = loss.mean()
    return result

if __name__ == "__main__":
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    torch_loss = F.binary_cross_entropy_with_logits(input, target)
    torch_loss.backward()
    print(f"Input: {input}")
    print(f"Target: {target}")
    print(f"PyTorch bce_with_logits: {torch_loss.item():.6f}")

    # Our implementation
    target_np = target.detach().numpy()
    input_np = input.detach().numpy()
    our_np_loss = bce_with_logits_np(target_np, input_np)
    print(f"Our numpy bce_with_logits:     {our_np_loss:.6f}")

    our_np_loss_stable = bce_with_logits_np_stable(target_np, input_np)
    print(f"Our numpy bce_with_logits stable: {our_np_loss_stable:.6f}")

    # PyTorch implementation
    our_torch_loss = bce_with_logits_torch(target, input)
    our_torch_loss.backward()
    print(f"Our torch bce_with_logits:     {our_torch_loss.item():.6f}")

    our_torch_loss_stable = bce_with_logits_torch_stable(target, input)
    our_torch_loss_stable.backward()
    print(f"Our torch bce_with_logits stable: {our_torch_loss_stable.item():.6f}")
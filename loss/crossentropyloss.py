# cross entropy loss

import torch
from torch.nn import functional as F
import numpy as np

# numpy implementation of softmax, log_softmax, cross_entropy
def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - m
    exp_i = np.exp(shifted_logits)
    exp_j = np.sum(exp_i, axis=1, keepdims=True)
    result: np.ndarray = exp_i / exp_j # softmax_i = exp_i / exp_j
    return result

def log_softmax_np_1(logits: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.log(softmax_np(logits)) # log_softmax_i = log(softmax_i)
    return result

def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - m
    exp_i = np.exp(shifted_logits)
    exp_j = np.sum(exp_i, axis=1, keepdims=True)
    result: np.ndarray = shifted_logits - np.log(exp_j) # log_softmax_i = yi - log(exp_j)
    return result

def cross_entropy_np(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Numpy implementation of multi-class cross entropy loss.
    y: class indices of shape (N,)
    y_pred: raw logits of shape (N, C)
    """
    log_probs = log_softmax_np(y_pred)
    row = np.arange(len(y))
    cols = y
    ce = -1 * log_probs[row, cols]
    result: float = ce.mean()
    return result


# pytorch implementation of softmax, log_softmax, cross_entropy
def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=1, keepdim=True).values # m = max(logits_i)
    shifted_logits = logits - m # shifted_logits_i = logits_i - m
    exp_i = torch.exp(shifted_logits) # exp_i = exp(shifted_logits_i)
    exp_j = torch.sum(exp_i, dim=1, keepdim=True) # exp_j = sum(exp_i)
    return exp_i / exp_j # softmax_i = exp_i / exp_j

def log_softmax_torch_1(logits: torch.Tensor) -> torch.Tensor:
    return torch.log(softmax_torch(logits)) # log_softmax_i = log(softmax_i)

def log_softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=1, keepdim=True).values # m = max(logits_i)
    shifted_logits = logits - m # shifted_logits_i = logits_i - m
    exp_i = torch.exp(shifted_logits) # exp_i = exp(shifted_logits_i)
    exp_j = torch.sum(exp_i, dim=1, keepdim=True) # exp_j = sum(exp_i)
    return shifted_logits - torch.log(exp_j) # log_softmax_i = yi - log(exp_j)

def cross_entropy_torch(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Pytorch implementation of multi-class cross entropy loss.
    y: class indices of shape (N,)
    y_pred: raw logits of shape (N, C)
    """
    log_probs = log_softmax_torch(y_pred)
    row = torch.arange(len(y))
    cols = y
    ce = -1 * log_probs[row, cols]
    return ce.mean()

# testing
if __name__ == "__main__":

    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    print(input)
    print(target)

    # PyTorch reference
    torch_loss = F.cross_entropy(input, target)
    torch_loss.backward()
    print(f"PyTorch cross_entropy: {torch_loss.item():.6f}")

    # Our implementation
    target_np = target.detach().numpy()
    input_np = input.detach().numpy()
    our_np_loss = cross_entropy_np(target_np, input_np)
    print(f"Our numpy cross_entropy:     {our_np_loss:.6f}")

    our_torch_loss = cross_entropy_torch(target, input)
    our_torch_loss.backward()
    print(f"Our torch cross_entropy:     {our_torch_loss.item():.6f}")
   
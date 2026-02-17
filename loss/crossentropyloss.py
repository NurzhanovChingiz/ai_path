# cross entropy loss

import torch
from torch.nn import functional as F
import numpy as np

# numpy implementation of softmax, log_softmax, cross_entropy
def softmax_np(logits):
    exp_i = np.exp(logits)
    exp_j = np.sum(exp_i, axis=1, keepdims=True)
    return exp_i / exp_j

def log_softmax_np(logits):
    return np.log(softmax_np(logits))

def cross_entropy_np(y, y_pred):
    """Multi-class cross entropy loss.
    y: class indices of shape (N,)
    y_pred: raw logits of shape (N, C)
    """
    log_probs = log_softmax_np(y_pred)
    
    return -1* np.sum(log_probs[np.arange(len(y)), y]) / y.shape[0]

# pytorch implementation of softmax, log_softmax, cross_entropy

def softmax_torch(logits):
    exp_i = torch.exp(logits)
    exp_j = torch.sum(exp_i, dim=1, keepdim=True)
    return exp_i / exp_j

def log_softmax_torch(logits):
    return torch.log(softmax_torch(logits))

def cross_entropy_torch(y, y_pred):
    return -1 * torch.sum(log_softmax_torch(y_pred)[np.arange(len(y)), y]) / y.shape[0]

# testing
if __name__ == "__main__":

    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    print(input)
    print(target)

    # PyTorch reference
    torch_loss = F.cross_entropy(input, target)
    print(f"PyTorch cross_entropy: {torch_loss.item():.6f}")

    # Our implementation
    target_np = target.detach().numpy()
    input_np = input.detach().numpy()
    our_np_loss = cross_entropy_np(target_np, input_np)
    print(f"Our numpy cross_entropy:     {our_np_loss:.6f}")

    our_torch_loss = cross_entropy_torch(target, input)
    print(f"Our torch cross_entropy:     {our_torch_loss.item():.6f}")
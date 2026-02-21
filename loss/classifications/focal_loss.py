# focal loss
# Where: single class, multi-class, multi-label,
# any setting with tons of easy negatives.
# Pros:
# good for imbalanced datasets
# more stabel than cross entropy when class imbalance
# more stabel than BCE with logits when class imbalance
# Cons:
# need tuning of alpha and gamma
# can be unstable in optimization
# can hurt probability calibration
# when use:
# detection/segmentation where background dominates

import numpy as np
import torch
from karina_focal_loss import focal_loss


def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - m
    exp_i = np.exp(shifted_logits)
    exp_j = np.sum(exp_i, axis=1, keepdims=True)
    result: np.ndarray = exp_i / exp_j  # softmax_i = exp_i / exp_j
    return result


def log_softmax_np_1(logits: np.ndarray) -> np.ndarray:
    # log_softmax_i = log(softmax_i)
    result: np.ndarray = np.log(softmax_np(logits))
    return result


def one_hot_np(
        labels: np.ndarray,
        num_classes: int,
        eps: float = 1e-10) -> np.ndarray:
    """Convert integer labels (N, *) to one-hot (N, C, *) with eps smoothing."""
    flat = labels.reshape(-1)
    one_hot = np.eye(num_classes, dtype=np.float32)[flat]  # (N*..., C)

    target_shape = labels.shape + (num_classes,)
    one_hot = one_hot.reshape(target_shape)

    ndim = labels.ndim
    perm = [0] + [ndim] + list(range(1, ndim))
    one_hot = one_hot.transpose(perm)

    result: np.ndarray = one_hot * (1.0 - eps) + eps
    return result


def focal_loss_np(
    pred: np.ndarray,
    target: np.ndarray,
    alpha: float,
    gamma: float,
) -> float:
    """
    Numpy implementation of focal loss matching Kornia's multi-class version.
    pred:   raw logits of shape (N, C, *)
    target: class indices of shape (N, *)
    alpha:  weight for classes 1..C-1 (class 0 gets weight 1-alpha)
    gamma:  focusing parameter
    """
    num_classes = pred.shape[1]

    target_one_hot = one_hot_np(target, num_classes)

    log_p = log_softmax_np_1(pred)

    loss = -np.power(1.0 - np.exp(log_p), gamma) * log_p * target_one_hot

    alpha_fac = np.array([1 - alpha] + [alpha] * (num_classes - 1))
    broadcast_shape = [-1] + [1] * len(pred.shape[2:])
    alpha_fac = alpha_fac.reshape(broadcast_shape)
    loss = alpha_fac * loss
    result: float = loss.mean()
    return result

# PyTorch implementation


def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=1, keepdim=True).values  # m = max(logits_i)
    shifted_logits = logits - m  # shifted_logits_i = logits_i - m
    exp_i = torch.exp(shifted_logits)  # exp_i = exp(shifted_logits_i)
    exp_j = torch.sum(exp_i, dim=1, keepdim=True)  # exp_j = sum(exp_i)
    return exp_i / exp_j  # softmax_i = exp_i / exp_j


def log_softmax_torch_1(logits: torch.Tensor) -> torch.Tensor:
    result: torch.Tensor = torch.log(softmax_torch(
        logits))  # log_softmax_i = log(softmax_i)
    return result


def one_hot_torch(
        labels: torch.Tensor,
        num_classes: int,
        eps: float = 1e-10) -> torch.Tensor:
    """Convert integer labels (N, *) to one-hot (N, C, *) with eps smoothing."""
    flat = labels.reshape(-1)
    one_hot = torch.eye(num_classes, dtype=torch.float32)[flat]  # (N*..., C)
    target_shape = labels.shape + (num_classes,)
    one_hot = one_hot.reshape(target_shape)
    ndim = labels.ndim
    perm = [0] + [ndim] + list(range(1, ndim))
    one_hot = one_hot.permute(perm)
    result: torch.Tensor = one_hot * (1.0 - eps) + eps
    return result


def focal_loss_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """
    PyTorch implementation of focal loss matching Kornia's multi-class version.
    """
    num_classes = pred.shape[1]
    target_one_hot = one_hot_torch(target, num_classes)
    log_p = log_softmax_torch_1(pred)
    loss = -torch.pow(1.0 - torch.exp(log_p), gamma) * log_p * target_one_hot
    alpha_fac = torch.tensor([1 - alpha] + [alpha] * (num_classes - 1))
    broadcast_shape = [-1] + [1] * len(pred.shape[2:])
    alpha_fac = alpha_fac.reshape(broadcast_shape)
    loss = alpha_fac * loss
    result: torch.Tensor = loss.mean()
    return result


if __name__ == "__main__":
    C = 3  # num_classes
    input = torch.randn(1, C, 3, 5, requires_grad=True)
    target = torch.randint(C, (1, 3, 5))
    alpha = 0.5
    gamma = 2.0
    loss = focal_loss(input, target, alpha, gamma, reduction='mean')
    print(f"Input: {input}")
    print(f"Target: {target}")
    print(f"Kornia focal loss: {loss.item():.6f}")

    target_np = target.detach().numpy()
    input_np = input.detach().numpy()
    our_loss = focal_loss_np(input_np, target_np, alpha, gamma)
    print(f"Our numpy focal loss: {our_loss:.6f}")
    # PyTorch implementation
    our_loss_torch = focal_loss_torch(input, target, alpha, gamma)
    print(f"Our torch focal loss: {our_loss_torch.item():.6f}")

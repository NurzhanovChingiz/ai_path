# SmoothL1 loss
# Where: bbox/keypoint coordinates, depth, optical flow, general regression
# Pros:
# L1 is more robust to outliers than MSE
# smoothL1/Huber is a common "sweet spot": stable gradients, less sensitive to outliers
# good for regression tasks
# good for bounding box regression
# Cons:
# can bias toward "median-ish" solutions; sometimes misses fine details.
# when use:
# strong baseline for regression; bbox deltas are often SmoothL1/Huber


import numpy as np
import torch
from torch.nn import functional as F

# numpy implementation


def smoothL1_np(
        pred: np.ndarray,
        target: np.ndarray,
        beta: float = 1.0) -> float:
    '''
    L1 (Huber) loss:
    Formula:
        smoothL1(x) = 0.5 * x^2 / beta, if |x| < beta
        smoothL1(x) = |x| - 0.5 * beta, otherwise
    Args:
        pred: predicted values
        target: target values
        beta: beta value

    Returns:
        smoothL1 loss as a float
    '''
    diff = np.abs(pred - target)
    mask = diff < beta
    result = np.where(
        mask, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return float(result.mean())

# pytorch implementation


def smoothL1_torch(
        pred: torch.Tensor,
        target: torch.Tensor,
        beta: float = 1.0) -> torch.Tensor:
    diff = torch.abs(pred - target)
    mask = diff < beta
    result: torch.Tensor = torch.where(
        mask, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return result.mean()


if __name__ == "__main__":
    pred = torch.randn(10, requires_grad=True)
    target = torch.randn(10)
    beta = 1.0
    print(f"Input: {pred}")
    print(f"Target: {target}")
    pred_np = pred.detach().numpy()
    target_np = target.detach().numpy()
    print(
        f"PyTorch smoothL1 loss: {
            F.smooth_l1_loss(
                pred,
                target,
                beta=beta):.6f}")
    print(f"Our numpy smoothL1 loss: {
        smoothL1_np(
            pred_np,
            target_np,
            beta):.6f}")
    print(f"Our torch smoothL1 loss: {smoothL1_torch(pred, target, beta):.6f}")

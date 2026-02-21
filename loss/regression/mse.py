# MSE loss
# Where: bbox/keypoint coordinates, depth, optical flow, general regression
# Pros:
# penalizes large errors strongly
# good for regression tasks
# Cons:
# sensitive to outliers
# can lead to "blurry averages" in image generation/reconstruction.
# when use:
# when noise is close to Gaussian and outliers are rare

import torch
from torch.nn import functional as F
import numpy as np
from typing import Optional

# numpy implementation


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    MSE loss:
    Formula:
        MSE(x) = 1/n * sum((x - y)^2)
    Args:
        pred: predicted values
        target: target values

    Returns:
        MSE loss as a float
    '''
    diff = pred - target
    diff_squared = diff ** 2
    result: np.ndarray = np.mean(diff_squared)
    return float(result)

# pytorch implementation


def mse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    diff_squared = diff ** 2
    result: torch.Tensor = torch.mean(diff_squared)
    return result


if __name__ == "__main__":
    pred = torch.randn(10, requires_grad=True)
    target = torch.randn(10)
    print(f"Input: {pred}")
    print(f"Target: {target}")
    pred_np = pred.detach().numpy()
    target_np = target.detach().numpy()
    print(f"PyTorch mse loss: {F.mse_loss(pred, target):.6f}")
    # our implementation
    print(f"Our numpy mse loss: {mse_np(pred_np, target_np):.6f}")
    print(f"Our torch mse loss: {mse_torch(pred, target):.6f}")

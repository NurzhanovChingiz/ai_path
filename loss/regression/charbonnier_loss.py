# Charbonnier loss
# Where: bbox/keypoint coordinates, depth, optical flow, general regression
# Pros:
# Robust + smooth optimization
# Cons:
# Not standard in core PyTorch; usually custom implementation.
# when use:
# image restoration tasks, super-resolution, denoising, etc.


import numpy as np
import torch
from korina_charbonnier_loss import charbonnier_loss

# numpy implementation


def charbonnier_loss_np(pred: np.ndarray, target: np.ndarray) -> float:
    """Charbonnier loss.

    Formula:
        Charbonnier(x) = ((pred - target) ** 2 + 1.0).sqrt() - 1.0
    Args:
        pred: predicted values
        target: target values

    Returns:
        Charbonnier loss as a float
    """
    diff = pred - target
    diff_squared = diff ** 2
    diff_squared_plus_1 = diff_squared + 1.0
    diff_squared_plus_1_sqrt = np.sqrt(diff_squared_plus_1)
    result: np.ndarray = np.mean(diff_squared_plus_1_sqrt - 1.0)
    return float(result)

# pytorch implementation


def charbonnier_loss_torch(
        pred: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    diff_squared = diff ** 2
    diff_squared_plus_1 = diff_squared + 1.0
    diff_squared_plus_1_sqrt = torch.sqrt(diff_squared_plus_1)
    result: torch.Tensor = torch.mean(diff_squared_plus_1_sqrt - 1.0)
    return result


if __name__ == "__main__":
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    print(f"Input: {pred}")
    print(f"Target: {target}")
    pred_np = pred.detach().numpy()
    target_np = target.detach().numpy()
    print(f"Kornia implementation charbonnier loss: {
        charbonnier_loss(
            pred,
            target,
            reduction="mean"):.6f}")
    # our implementation
    print(
        f"Our numpy charbonnier loss: {
            charbonnier_loss_np(
                pred_np,
                target_np):.6f}")
    print(
        f"Our torch charbonnier loss: {
            charbonnier_loss_torch(
                pred,
                target):.6f}")

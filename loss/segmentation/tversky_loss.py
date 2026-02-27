"""Tversky loss for segmentation with adjustable FN/FP trade-off."""
# Tversky loss for segmentation
# segmentation where you want to bias toward "don't miss positives" vs "don't add false positives"
# Pros:
# adjustable FN/FP trade-off via parameters
# Cons:
# more hyperparameters to tune.
# when use:
# medical segmentation, tiny objects, FN-critical scenarios


import numpy as np
import torch

from loss.segmentation.karina_tversky_loss import tversky_loss

# numpy implementation


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Calculate the softmax of the logits.

    Args:
        logits: The logits.

    Returns:
        The softmax of the logits.
    """
    m = logits.max(axis=1, keepdims=True)
    shifted_logits = logits - m
    exp_i = np.exp(shifted_logits)
    exp_j = np.sum(exp_i, axis=1, keepdims=True)
    result: np.ndarray = exp_i / exp_j  # softmax_i = exp_i / exp_j
    return result


def tversky_loss_np(
    pred: np.ndarray,
    target: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
) -> np.ndarray:
    """Calculate the Tversky loss.

    Args:
        pred: The predicted logits.
        target: The target tensor.
        alpha: The alpha parameter.
        beta: The beta parameter.
        eps: The epsilon parameter.
    """
    pred_soft = softmax_np(pred)
    p_true = np.take_along_axis(
        pred_soft, target[:, None, :, :], axis=1).squeeze(1)  # (B, H, W)
    intersection = p_true.sum(axis=(1, 2))
    total = p_true.shape[1] * p_true.shape[2]
    denominator = intersection + (alpha + beta) * (total - intersection) + eps
    score = intersection / denominator
    result: np.ndarray = 1.0 - score.mean()
    return result

# pytorch implementation


def tversky_loss_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Calculate the Tversky loss.

    Args:
        pred: The predicted logits.
        target: The target tensor.
        alpha: The alpha parameter.
        beta: The beta parameter.
        eps: The epsilon parameter.
    """
    pred_soft = torch.nn.functional.softmax(pred, dim=1)
    p_true = torch.gather(
        pred_soft, 1, target[:, None, :, :]).squeeze(1)  # (B, H, W)
    intersection = p_true.sum(dim=(1, 2))
    total = p_true.shape[1] * p_true.shape[2]
    denominator = intersection + (alpha + beta) * (total - intersection) + eps
    score = intersection / denominator
    return 1.0 - score.mean()


if __name__ == "__main__":
    N = 5  # num_classes
    pred = torch.randn(1, N, 3, 5, requires_grad=True)
    target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
    print(f"Input: {pred}")
    print(f"Target: {target}")
    alpha = 0.5
    beta = 0.5
    eps = 1e-8
    ignore_index = -100
    loss = tversky_loss(pred, target, alpha, beta, eps, ignore_index)
    print(f"Tversky loss: {loss.item():.6f}")

    # Our implementation
    pred_np = pred.detach().numpy()
    target_np = target.detach().numpy()
    loss_np = tversky_loss_np(
        pred_np,
        target_np,
        alpha,
        beta,
        eps,
    )
    print(f"Our numpy tversky loss: {loss_np:.6f}")

    loss_torch = tversky_loss_torch(
        pred, target, alpha, beta, eps)
    print(f"Our torch tversky loss: {loss_torch.item():.6f}")

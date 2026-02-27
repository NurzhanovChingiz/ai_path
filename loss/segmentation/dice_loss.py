"""Dice loss for binary and multi-class segmentation."""
# dice loss for segmentation
# Where: binary or multi-class segmentation,
# especially common in medical imaging.
# Pros:
# directly optimizes overlap between predicted and target
# better than cross entropy loss in imbalance datasets
# # often better than pure CE under area imbalances, but worse than Focal Loss
# Cons:
# can be unstable with empty masks
# can produce weaker boundaries / thin structures unless combined with CE/BCE or boundary-aware losses.
# when use:
# when overlap is the main goal under imbalance
import numpy as np
import torch

from loss.segmentation.karina_dice_loss import dice_loss

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


def one_hot_np(
        labels: np.ndarray,
        num_classes: int,
        eps: float = 1e-10) -> np.ndarray:
    """Convert integer labels (N, *) to one-hot (N, C, *) with eps smoothing.

    Args:
        labels: The labels.
        num_classes: The number of classes.
        eps: The epsilon parameter.

    Returns:
        The one-hot encoded labels.
    """
    flat = labels.reshape(-1)
    one_hot = np.eye(num_classes, dtype=np.float32)[flat]  # (N*..., C)

    target_shape = (*labels.shape, num_classes)
    one_hot = one_hot.reshape(target_shape)

    ndim = labels.ndim
    perm = [0, ndim, *list(range(1, ndim))]
    one_hot = one_hot.transpose(perm)

    result: np.ndarray = one_hot * (1.0 - eps) + eps
    return result


def dice_loss_np(
        pred: np.ndarray,
        target: np.ndarray,
        smooth: float = 1) -> float:
    """Calculate the Dice loss.

    Args:
        pred: The predicted logits.
        target: The target tensor.
        smooth: The smooth parameter.
    """
    pred = softmax_np(pred)
    target_one_hot = one_hot_np(target, num_classes=pred.shape[1])
    intersection = np.sum(pred * target_one_hot)
    union = np.sum(pred) + np.sum(target_one_hot)
    dice: np.ndarray = 1 - 2 * intersection / (union + smooth)
    result: float = float(dice.mean())
    return result

# PyTorch implementation


def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    """Calculate the softmax of the logits.

    Args:
        logits: The logits.

    Returns:
        The softmax of the logits.
    """
    m = logits.max(dim=1, keepdim=True).values  # m = max(logits_i)
    shifted_logits = logits - m  # shifted_logits_i = logits_i - m
    exp_i = torch.exp(shifted_logits)  # exp_i = exp(shifted_logits_i)
    exp_j = torch.sum(exp_i, dim=1, keepdim=True)  # exp_j = sum(exp_i)
    return exp_i / exp_j  # softmax_i = exp_i / exp_j


def one_hot_torch(
        labels: torch.Tensor,
        num_classes: int,
        eps: float = 1e-10) -> torch.Tensor:
    """Convert integer labels (N, *) to one-hot (N, C, *) with eps smoothing.

    Args:
        labels: The labels.
        num_classes: The number of classes.
        eps: The epsilon parameter.

    Returns:
        The one-hot encoded labels.
    """
    flat = labels.reshape(-1)
    one_hot = torch.eye(num_classes, dtype=torch.float32)[flat]  # (N*..., C)
    target_shape = (*labels.shape, num_classes)
    one_hot = one_hot.reshape(target_shape)
    ndim = labels.ndim
    perm = [0, ndim, *list(range(1, ndim))]
    one_hot = one_hot.permute(perm)
    result: torch.Tensor = one_hot * (1.0 - eps) + eps
    return result


def dice_loss_torch(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1) -> torch.Tensor:
    """Calculate the Dice loss.

    Args:
        pred: The predicted logits.
        target: The target tensor.
        smooth: The smooth parameter.
    """
    pred = softmax_torch(pred)
    target_one_hot = one_hot_torch(target, num_classes=pred.shape[1])
    intersection = torch.sum(pred * target_one_hot)
    union = torch.sum(pred) + torch.sum(target_one_hot)
    dice: torch.Tensor = 1 - 2 * intersection / (union + smooth)
    result: torch.Tensor = dice.mean()
    return result


if __name__ == "__main__":
    N = 5  # num_classes
    x = torch.randn(1, N, 3, 5, requires_grad=True)
    target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
    print(f"Input: {x}")
    print(f"Target: {target}")
    smooth = 1e-8

    target_np = target.detach().numpy()
    x_np = x.detach().numpy()

    torch_loss = dice_loss(x, target, eps=smooth, average="micro")
    torch_loss.backward()
    print(f"Karina implementation dice loss: {torch_loss.item():.6f}")

   # Our implementation
    print(
        f"Our numpy dice loss: {
            dice_loss_np(
                x_np,
                target_np,
                smooth):.6f}")
    print(f"Our torch dice loss: {dice_loss_torch(x, target, smooth):.6f}")

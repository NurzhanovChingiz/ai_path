# IoU for segmentation
# Where: binary or multi-class segmentation,
# Pros:
# easy to understand and interpret
# directly targets IoU.
# Cons:
# sensitive to class imbalance
# rare classes and empty masks can dominate
# when use:
# when evaluation is IoU and you want to push that specifically
import numpy as np
import torch


# numpy implementation

def softmax_np(logits: np.ndarray) -> np.ndarray:
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


def iou_np(pred: np.ndarray, target: np.ndarray, smooth: float = 1) -> float:
    pred = softmax_np(pred)
    target_one_hot = one_hot_np(target, num_classes=pred.shape[1])
    intersection = np.sum(pred * target_one_hot)
    sum_of_cardinalities = np.sum(pred) + np.sum(target_one_hot)
    iou: np.ndarray = 1 - intersection / \
        (sum_of_cardinalities - intersection + smooth)
    result: float = float(iou.mean())
    return result

# PyTorch implementation


def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=1, keepdim=True).values
    shifted_logits = logits - m
    exp_i = torch.exp(shifted_logits)
    exp_j = torch.sum(exp_i, dim=1, keepdim=True)
    return exp_i / exp_j


def one_hot_torch(
        labels: torch.Tensor,
        num_classes: int,
        eps: float = 1e-10) -> torch.Tensor:
    """Convert integer labels (N, *) to one-hot (N, C, *) with eps smoothing."""
    flat = labels.reshape(-1)
    one_hot = torch.eye(num_classes, dtype=torch.float32)[flat]
    target_shape = labels.shape + (num_classes,)
    one_hot = one_hot.reshape(target_shape)
    ndim = labels.ndim
    perm = [0] + [ndim] + list(range(1, ndim))
    one_hot = one_hot.permute(perm)
    result: torch.Tensor = one_hot * (1.0 - eps) + eps
    return result


def iou_loss_torch(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1) -> torch.Tensor:
    pred = softmax_torch(pred)
    target_one_hot = one_hot_torch(target, num_classes=pred.shape[1])
    intersection = torch.sum(pred * target_one_hot)
    sum_of_cardinalities = torch.sum(pred) + torch.sum(target_one_hot)
    iou: torch.Tensor = 1 - intersection / \
        (sum_of_cardinalities - intersection + smooth)
    result: torch.Tensor = iou.mean()
    return result


if __name__ == "__main__":
    N = 5  # num_classes
    input = torch.randn(1, N, 3, 5, requires_grad=True)
    target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
    print(f"Input: {input}")
    print(f"Target: {target}")
    smooth = 1e-8

    target_np = target.detach().numpy()
    input_np = input.detach().numpy()

    print(f"Our numpy iou loss: {iou_np(input_np, target_np, smooth):.6f}")
    torch_iou = iou_loss_torch(input, target, smooth)
    torch_iou.backward()
    print(f"Our torch iou loss: {torch_iou.item():.6f}")

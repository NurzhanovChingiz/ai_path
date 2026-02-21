# dice loss for segmentation
# Where: binary or multi-class segmentation,
# especially common in medical imaging.
# Pros:
# directly optimizes overlap between predicted and target
# better than cross entropy loss in disbalance datasets
# # often better than pure CE under area imbalances, but worse than Focal Loss
# Cons:
# can be unstable with empty masks
# can produce weaker boundaries / thin structures unless combined with CE/BCE or boundary-aware losses.
# when use: 
# when overlap is the main goal under imbalance
import numpy as np
import torch
from karina_dice_loss import dice_loss

# numpy implementation

def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    shifted_logits = logits - m
    exp_i = np.exp(shifted_logits)
    exp_j = np.sum(exp_i, axis=1, keepdims=True)
    result: np.ndarray = exp_i / exp_j # softmax_i = exp_i / exp_j
    return result

def one_hot_np(labels: np.ndarray, num_classes: int, eps: float = 1e-10) -> np.ndarray:
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

def dice_loss_np(pred: np.ndarray, target: np.ndarray, smooth: float = 1) -> float:
    pred = softmax_np(pred)
    target_one_hot = one_hot_np(target, num_classes=pred.shape[1])
    intersection = np.sum(pred * target_one_hot)
    union = np.sum(pred) + np.sum(target_one_hot)
    dice: np.ndarray = 1 - 2 * intersection / (union + smooth)
    result: float = float(dice.mean())
    return result

# PyTorch implementation
def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=1, keepdim=True).values # m = max(logits_i)
    shifted_logits = logits - m # shifted_logits_i = logits_i - m
    exp_i = torch.exp(shifted_logits) # exp_i = exp(shifted_logits_i)
    exp_j = torch.sum(exp_i, dim=1, keepdim=True) # exp_j = sum(exp_i)
    return exp_i / exp_j # softmax_i = exp_i / exp_j

def one_hot_torch(labels: torch.Tensor, num_classes: int, eps: float = 1e-10) -> torch.Tensor:
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

def dice_loss_torch(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
    pred = softmax_torch(pred)
    target_one_hot = one_hot_torch(target, num_classes=pred.shape[1])
    intersection = torch.sum(pred * target_one_hot)
    union = torch.sum(pred) + torch.sum(target_one_hot)
    dice: torch.Tensor = 1 - 2 * intersection / (union + smooth)
    result: torch.Tensor = dice.mean()
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

    torch_loss = dice_loss(input, target, eps=smooth, average='micro')
    torch_loss.backward()
    print(f"Karina implementation dice loss: {torch_loss.item():.6f}")

   # Our implementation
    print(f"Our numpy dice loss: {dice_loss_np(input_np, target_np, smooth):.6f}")
    print(f"Our torch dice loss: {dice_loss_torch(input, target, smooth):.6f}")
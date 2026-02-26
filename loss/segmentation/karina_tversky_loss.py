# Karina implementation of Tversky loss for segmentation
# from:
# https://github.com/kornia/kornia/blob/0f8d1972603ed10f549c66c9613669f886046b23/kornia/losses/tversky.py

import torch
from torch.nn import functional as F


def mask_ignore_pixels(
    target: torch.Tensor, ignore_index: int | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Mask the ignore pixels.

    Args:
        target: The target tensor.
        ignore_index: The ignore index.

    Returns:
        The target tensor and the target mask.
    """
    if ignore_index is None:
        return target, None

    target_mask = target != ignore_index

    if target_mask.all():
        return target, None

    # map invalid pixels to a valid class (0)
    # they need to be manually excluded from the loss computation after
    target = target.where(target_mask, target.new_zeros(1))

    return target, target_mask


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-8,
    ignore_index: int | None = -100,
) -> torch.Tensor:
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels tensor with shape :math:`(N, H, W)` where each value
          is in range :math:`0 ≤ targets[i] ≤ C-1`.
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = tversky_loss(pred, target, alpha=0.5, beta=0.5)
        >>> output.backward()

    """
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"pred type is not a torch.Tensor. Got {type(pred)}")

    if not len(pred.shape) == 4:
        raise ValueError(
            f"Invalid pred shape, we expect BxNxHxW. Got: {
                pred.shape}")

    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(
            f"pred and target shapes must be the same. Got: {
                pred.shape} and {
                target.shape}")

    if not pred.device == target.device:
        raise ValueError(
            f"pred and target must be in the same device. Got: {
                pred.device} and {
                target.device}")

    # compute softmax over the classes axis
    pred_soft = F.softmax(pred, dim=1)
    target, target_mask = mask_ignore_pixels(target, ignore_index)

    p_true = pred_soft.gather(1, target.unsqueeze(1))  # (B,1,H,W)

    if target_mask is not None:
        m = target_mask.unsqueeze(1).to(dtype=pred.dtype)
        p_true = p_true * m
        total = m.sum((1, 2, 3))
    else:
        B, _, H, W = pred.shape
        total = torch.full((B,), H * W, dtype=pred.dtype, device=pred.device)

    intersection = p_true.sum((1, 2, 3))
    # denominator = intersection + (alpha + beta) * (total - intersection) + eps
    # instead of multiple ops, do it in one fused step:
    denominator = torch.addcmul(
        intersection,  # base
        total - intersection,  # tensor1
        torch.full_like(total, alpha + beta),  # tensor2 (scalar as tensor)
        value=1.0,  # (intersection) + 1 * (tensor1*tensor2)
    ).add_(eps)  # in-place add eps
    score = intersection.div(denominator)

    return 1.0 - score.mean()

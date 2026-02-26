"""Kornia-style dice loss implementation."""
# Karina implementation of dice loss for segmentation
# from
# https://github.com/kornia/kornia/blob/0f8d1972603ed10f549c66c9613669f886046b23/kornia/losses/dice.py
from typing import Any

import torch
from torch.nn import functional as F

_KORNIA_CHECKS_ENABLED: bool = True


class BaseError(Exception):
    """Base exception class for all Kornia errors."""

    pass


class TypeCheckError(BaseError):
    """Raised when type validation fails.

    Attributes:
        actual_type: The actual type that failed validation.
        expected_type: The expected type.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_type: type | None = None,
        expected_type: type | tuple[type, ...] | None = None,
    ):
        """Initialize the TypeCheckError.

        Args:
            message: The message.
            actual_type: The actual type.
            expected_type: The expected type.
        """
        super().__init__(message)
        self.actual_type = actual_type
        self.expected_type = expected_type


def KORNIA_CHECK_IS_TENSOR(
        x: Any,
        msg: str | None = None,
        raises: bool = True) -> bool:
    """Check the input variable is a Tensor.

    Args:
        x: any input variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeCheckError: if the input variable does not match with the expected and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_TENSOR(x, "Invalid tensor")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not isinstance(x, torch.Tensor):
        if raises:
            # JIT doesn't support try-except or type introspection, so use
            # simple message
            if torch.jit.is_scripting():
                error_msg = "Type mismatch: expected Tensor."
                if msg is not None:
                    error_msg += f"\n  {msg}"
                raise TypeCheckError(error_msg)
            else:
                # In Python mode, we can safely use type introspection
                type_name = str(type(x))
                error_msg = f"Type mismatch: expected Tensor, got {type_name}."
                if msg is not None:
                    error_msg += f"\n  {msg}"
                raise TypeCheckError(
                    error_msg,
                    actual_type=type(x),
                    expected_type=torch.Tensor,
                )
        return False
    return True


def KORNIA_CHECK(
        condition: bool,
        msg: str | None = None,
        raises: bool = True) -> bool:
    """Check any arbitrary boolean condition.

    Args:
        condition: the condition to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        BaseError: if the condition is not met and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK(x.shape[-2:] == (3, 3), "Invalid homography")
        True

    """
    if not torch.jit.is_scripting():
        if not _KORNIA_CHECKS_ENABLED:
            return True

    if not condition:
        if raises:
            if msg is None:
                error_msg = "Validation condition failed"
            else:
                error_msg = msg
            raise BaseError(error_msg)
        return False
    return True


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


def one_hot(
        labels: torch.Tensor,
        num_classes: int,
        device: torch.device,
        dtype: torch.dtype,
        eps: float = 1e-6) -> torch.Tensor:
    r"""Convert an integer label x-D torch.Tensor to a one-hot (x+1)-D torch.Tensor.

    Args:
        labels: torch.Tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned torch.Tensor.
        dtype: the desired data type of returned torch.Tensor.
        eps: epsilon for numerical stability.

    Returns:
        the labels in one hot torch.Tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3, device=torch.device('cpu'), dtype=torch.float32)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    KORNIA_CHECK_IS_TENSOR(labels, "Input labels must be a torch.Tensor")
    KORNIA_CHECK(
        labels.dtype == torch.int64,
        f"labels must be of dtype torch.int64. Got: {
            labels.dtype}")
    KORNIA_CHECK(
        num_classes >= 1,
        f"The number of classes must be >= 1. Got: {num_classes}")

    # Use PyTorch's built-in one_hot function
    one_hot_tensor = F.one_hot(labels, num_classes=num_classes)

    # PyTorch's one_hot adds the class dimension at the end: (*, num_classes)
    # We need it at position 1: (N, C, *)
    # Permute: move the last dimension (num_classes) to position 1
    ndim = labels.ndim
    permute_dims = [0] + [ndim] + list(range(1, ndim))
    one_hot_tensor = one_hot_tensor.permute(*permute_dims)

    # Convert to desired dtype and device, then apply eps for numerical
    # stability
    one_hot_tensor = one_hot_tensor.to(dtype=dtype, device=device)
    # Apply eps: multiply by (1-eps) and add eps to all elements
    one_hot_tensor = one_hot_tensor * (1.0 - eps) + eps

    return one_hot_tensor


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    average: str = "micro",
    eps: float = 1e-8,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = -100,
) -> torch.Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X \cap Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot torch.Tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        pred: logits torch.Tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels torch.Tensor with shape :math:`(N, H, W)` where each value
          is in range :math:`0 ≤ targets[i] ≤ C-1`.
        average:
            Reduction applied in multi-class scenario:
            - ``'micro'`` [default]: Calculate the loss across all classes.
            - ``'macro'``: Calculate the loss for each class separately and average the metrics across classes.
        eps: Scalar to enforce numerical stabiliy.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        One-element torch.Tensor of the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = dice_loss(pred, target)
        >>> output.backward()

    """
    KORNIA_CHECK_IS_TENSOR(pred)

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
    num_of_classes = pred.shape[1]
    possible_average = {"micro", "macro"}
    KORNIA_CHECK(
        average in possible_average,
        f"The `average` has to be one of {possible_average}. Got: {average}")

    # compute F.softmax over the classes axis
    pred_soft: torch.Tensor = F.softmax(pred, dim=1)

    target, target_mask = mask_ignore_pixels(target, ignore_index)

    # create the labels one hot torch.Tensor
    target_one_hot: torch.Tensor = one_hot(
        target,
        num_classes=pred.shape[1],
        device=pred.device,
        dtype=pred.dtype)

    # mask ignore pixels
    if target_mask is not None:
        target_mask.unsqueeze_(1)
        target_one_hot = target_one_hot * target_mask
        pred_soft = pred_soft * target_mask

    # compute the actual dice score
    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be torch.Tensor or None.")
        KORNIA_CHECK(
            (weight.shape[0] == num_of_classes and weight.numel() == num_of_classes),
            f"weight shape must be (num_of_classes,): ({num_of_classes},), got {weight.shape}",
        )
        KORNIA_CHECK(
            weight.device == pred.device,
            f"weight and pred must be in the same device. Got: {
                weight.device} and {
                pred.device}",
        )
    else:
        weight = pred.new_ones(pred.shape[1])

    # set dimensions for the appropriate averaging
    dims: tuple[int, ...] = (2, 3)

    if average == "micro":
        dims = (1, *dims)

        weight = weight.view(-1, 1, 1)
        pred_soft = pred_soft * weight
        target_one_hot = target_one_hot * weight

    intersection = torch.sum(pred_soft * target_one_hot, dims)
    cardinality = torch.sum(pred_soft + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)
    dice_loss = -dice_score + 1.0

    # reduce the loss across samples (and classes in case of `macro` averaging)
    if average == "macro":
        dice_loss = (dice_loss * weight).sum(-1) / weight.sum()

    dice_loss = torch.mean(dice_loss)

    return dice_loss

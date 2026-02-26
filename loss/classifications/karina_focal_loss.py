"""Kornia-style focal loss implementation."""
from typing import Any

import torch
from torch.nn import functional as F

# Kornia implementation
# from
# https://github.com/kornia/kornia/tree/0f8d1972603ed10f549c66c9613669f886046b23
_KORNIA_CHECKS_ENABLED: bool = True


class BaseError(Exception):
    """Base exception class for all Kornia errors."""



class ShapeError(BaseError):
    """Raised when tensor shape validation fails.

    Attributes:
        actual_shape: The actual shape of the tensor that failed validation.
        expected_shape: The expected shape specification.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_shape: tuple[int, ...] | list[int] | None = None,
        expected_shape: list[str] | tuple[int, ...] | None = None,
    ) -> None:
        """Initialize ShapeError with message and optional shape attributes."""
        super().__init__(message)
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape


def KORNIA_CHECK_SHAPE(
        x: torch.Tensor,
        shape: list[str],
        msg: str | None = None,
        raises: bool = True) -> bool:
    """Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Args:
        x: the tensor to evaluate.
        shape: a list with strings with the expected shape.
        msg: optional custom message to append to error.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ShapeError: if the input tensor does not have the expected shape and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])  # implicit
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["2", "3", "H", "W"])  # explicit
        True

    """
    if not torch.jit.is_scripting() and not _KORNIA_CHECKS_ENABLED:
        return True

    if shape[0] == "*":
        shape_to_check = shape[1:]
        x_shape_to_check = x.shape[-len(shape) + 1:]
    elif shape[-1] == "*":
        shape_to_check = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            expected_dims = len(shape_to_check)
            actual_dims = len(x_shape_to_check)
            error_msg = f"Shape dimension mismatch: expected {expected_dims} dimensions, got {actual_dims}.\n"
            error_msg += f"  Expected shape: {shape}\n"
            x_shape_list = list(x.shape)
            error_msg += f"  Actual shape: {x_shape_list}"
            if msg is not None:
                error_msg += f"\n  {msg}"
            raise ShapeError(
                error_msg,
                actual_shape=x_shape_list,
                expected_shape=shape,
            )
        else:
            return False

    for i in range(len(x_shape_to_check)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            if raises:
                error_msg = f"Shape mismatch at dimension {i}: expected {dim}, got {
                    x_shape_to_check[i]}.\n"
                error_msg += f"  Expected shape: {shape}\n"
                x_shape_list = list(x.shape)
                error_msg += f"  Actual shape: {x_shape_list}"
                if msg is not None:
                    error_msg += f"\n  {msg}"
                raise ShapeError(
                    error_msg,
                    actual_shape=x_shape_list,
                    expected_shape=shape,
                )
            else:
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
    if not torch.jit.is_scripting() and not _KORNIA_CHECKS_ENABLED:
        return True

    if not condition:
        if raises:
            error_msg = "Validation condition failed" if msg is None else msg
            raise BaseError(error_msg)
        return False
    return True


def mask_ignore_pixels(
    target: torch.Tensor, ignore_index: int | None,
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
    ) -> None:
        """Initialize TypeCheckError with message and optional type attributes."""
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
    if not torch.jit.is_scripting() and not _KORNIA_CHECKS_ENABLED:
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
            labels.dtype}",
    )
    KORNIA_CHECK(
        num_classes >= 1,
        f"The number of classes must be >= 1. Got: {num_classes}")

    # Use PyTorch's built-in one_hot function
    one_hot_tensor = F.one_hot(labels, num_classes=num_classes)

    # PyTorch's one_hot adds the class dimension at the end: (*, num_classes)
    # We need it at position 1: (N, C, *)
    # Permute: move the last dimension (num_classes) to position 1
    ndim = labels.ndim
    permute_dims = [0, ndim, *list(range(1, ndim))]
    one_hot_tensor = one_hot_tensor.permute(*permute_dims)

    # Convert to desired dtype and device, then apply eps for numerical
    # stability
    one_hot_tensor = one_hot_tensor.to(dtype=dtype, device=device)
    # Apply eps: multiply by (1-eps) and add eps to all elements
    one_hot_tensor = one_hot_tensor * (1.0 - eps) + eps

    return one_hot_tensor


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float | None,
    gamma: float = 2.0,
    reduction: str = "none",
    weight: torch.Tensor | None = None,
    ignore_index: int | None = -100,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        pred: logits torch.Tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels torch.Tensor with shape :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        the computed loss.

    Example:
        >>> C = 5  # num_classes
        >>> pred = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> output = focal_loss(pred, target, **kwargs)
        >>> output.backward()

    """
    KORNIA_CHECK_SHAPE(pred, ["B", "C", "*"])
    out_size = (pred.shape[0], *pred.shape[2:])
    KORNIA_CHECK(
        (pred.shape[0] == target.shape[0] and target.shape[1:] == pred.shape[2:]),
        f"Expected target size {out_size}, got {target.shape}",
    )
    KORNIA_CHECK(
        pred.device == target.device,
        f"pred and target must be in the same device. Got: {
            pred.device} and {
            target.device}",
    )

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

    # compute F.softmax over the classes axis
    log_pred_soft: torch.Tensor = pred.log_softmax(1)

    # compute the actual focal loss
    loss_tmp: torch.Tensor = - \
        torch.pow(1.0 - log_pred_soft.exp(), gamma) * log_pred_soft * target_one_hot

    num_of_classes = pred.shape[1]
    broadcast_dims = [-1] + [1] * len(pred.shape[2:])
    if alpha is not None:
        alpha_fac = torch.tensor([1 -
                                  alpha] +
                                 [alpha] *
                                 (num_of_classes -
                                  1), dtype=loss_tmp.dtype, device=loss_tmp.device)
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss_tmp = alpha_fac * loss_tmp

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

        weight = weight.view(broadcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        msg = f"Invalid reduction mode: {reduction}"
        raise NotImplementedError(msg)
    return loss

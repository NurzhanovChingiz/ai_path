"""Kornia-style Charbonnier loss for regression."""
# Kornia implementation of charbonnier loss for regression
# from
# https://github.com/kornia/kornia/blob/0f8d1972603ed10f549c66c9613669f886046b23/kornia/losses/charbonnier.py

from typing import Any

import torch

_KORNIA_CHECKS_ENABLED = True


class BaseError(Exception):
    """Base exception class for all Kornia errors."""

    pass


class DeviceError(BaseError):
    """Raised when device mismatch validation fails.

    Attributes:
        actual_devices: The actual device(s) that failed validation.
        expected_device: The expected device.
    """

    def __init__(
        self,
        message: str,
        *,
        actual_devices: list | None = None,
        expected_device: Any | None = None,
    ) -> None:
        """Initialize DeviceError with message and optional device attributes."""
        super().__init__(message)
        self.actual_devices = actual_devices
        self.expected_device = expected_device


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
        expected_shape: list[str] | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Initialize ShapeError with message and optional shape attributes."""
        super().__init__(message)
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape


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


def KORNIA_CHECK_SAME_DEVICE(
        x: torch.Tensor,
        y: torch.Tensor,
        raises: bool = True) -> bool:
    """Check whether two tensor in the same device.

    Args:
        x: first tensor to evaluate.
        y: second tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        DeviceError: if the two tensors are not in the same device and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICE(x1, x2)
        True

    """
    if not torch.jit.is_scripting() and not _KORNIA_CHECKS_ENABLED:
        return True

    if x.device != y.device:
        if raises:
            error_msg = "Device mismatch: tensors must be on the same device.\n"
            error_msg += f"  First tensor device: {x.device}\n"
            error_msg += f"  Second tensor device: {y.device}"
            raise DeviceError(
                error_msg,
                actual_devices=[x.device, y.device],
                expected_device=x.device,
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
    if not torch.jit.is_scripting() and not _KORNIA_CHECKS_ENABLED:
        return True

    if not condition:
        if raises:
            error_msg = "Validation condition failed" if msg is None else msg
            raise BaseError(error_msg)
        return False
    return True


def KORNIA_CHECK_SAME_SHAPE(
        x: torch.Tensor,
        y: torch.Tensor,
        raises: bool = True) -> bool:
    """Check whether two tensor have the same shape.

    Args:
        x: first tensor to evaluate.
        y: second tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        ShapeError: if the two tensors have not the same shape and raises is True.

    Note:
        Checks can be disabled in Python mode using `disable_checks()` or the KORNIA_CHECKS
        environment variable. In TorchScript-compiled code, checks always run (TorchScript
        cannot access module-level globals, but the validation logic is fast). When running
        with `python -O`, Python's optimizer may eliminate some checks.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_SAME_SHAPE(x1, x2)
        True

    """
    if not torch.jit.is_scripting() and not _KORNIA_CHECKS_ENABLED:
        return True

    if x.shape != y.shape:
        if raises:
            error_msg = "Shape mismatch: tensors must have the same shape.\n"
            x_shape_list = list(x.shape)
            y_shape_list = list(y.shape)
            error_msg += f"  First tensor shape: {x_shape_list}\n"
            error_msg += f"  Second tensor shape: {y_shape_list}"
            raise ShapeError(
                error_msg,
                actual_shape=x_shape_list,
                expected_shape=y_shape_list,
            )
        return False
    return True


def charbonnier_loss(
        img1: torch.Tensor,
        img2: torch.Tensor,
        reduction: str = "none") -> torch.Tensor:
    r"""Criterion that computes the Charbonnier [2] (aka. L1-L2 [3]) loss.

    According to [1], we compute the Charbonnier loss as follows:

    .. math::

        \text{WL}(x, y) = \sqrt{(x - y)^{2} + 1} - 1

    Where:
       - :math:`x` is the prediction.
       - :math:`y` is the target to be regressed to.

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] https://ieeexplore.ieee.org/document/413553
        [3] https://hal.inria.fr/inria-00074015/document
        [4] https://arxiv.org/pdf/1712.05927.pdf

    .. note::
        This implementation follows the formulation by Barron [1]. Other works utilize
        a slightly different implementation (see [4]).

    Args:
        img1: the predicted torch.Tensor with shape :math:`(*)`.
        img2: the target torch.Tensor with the same shape as img1.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Return:
        a scalar with the computed loss.

    Example:
        >>> img1 = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 32)
        >>> output = charbonnier_loss(img1, img2, reduction="sum")
        >>> output.backward()

    """
    KORNIA_CHECK_IS_TENSOR(img1)

    KORNIA_CHECK_IS_TENSOR(img2)

    KORNIA_CHECK_SAME_SHAPE(img1, img2)

    KORNIA_CHECK_SAME_DEVICE(img1, img2)

    KORNIA_CHECK(reduction in ("mean", "sum", "none", None),
                 f"Given type of reduction is not supported. Got: {reduction}")

    # compute loss
    loss = ((img1 - img2) ** 2 + 1.0).sqrt() - 1.0

    # perform reduction
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none" or reduction is None:
        pass
    else:
        msg = "Invalid reduction option."
        raise NotImplementedError(msg)

    return loss

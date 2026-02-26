"""Tests for custom activation functions."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from activation.activation import (
    CustomCELU,
    CustomELU,
    CustomGELU,
    CustomLeakyReLU,
    CustomMish,
    CustomPReLU,
    CustomReLU,
    CustomReLU6,
    CustomSELU,
    CustomSigmoid,
    CustomSoftmax,
    CustomSoftplus,
    CustomSwish,
    CustomTanh,
)

ATOL: float = 1e-5  # 0.00001
RTOL: float = 1e-3  # 0.01

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def inputs_tensor() -> list[torch.Tensor]:
    """Return a list of tensors for activation function testing."""
    torch.manual_seed(42)
    inputs: list[torch.Tensor] = [
        torch.randn(2, 3, 4),
        torch.randn(8, 16),
        torch.randn(2, 3, 4) * 100,
        torch.randn(1, 1),
        torch.randn(1, 1, 1),
        torch.randn(1, 1) * -1,
        torch.randn(1, 1, 1) * 0,
        torch.randn(10, 10) * float('inf'),
        torch.randn(10, 10, 10) * float('-inf'),
        torch.randn(10, 10) * float('nan')
    ]
    return inputs
# ── Assertions ──────────────────────────────────────────────────────────


def assert_close_all(
        custom_out: torch.Tensor,
        ref_out: torch.Tensor,
        atol: float = ATOL,
        rtol: float = RTOL) -> None:
    """Assert two tensors are close using np.allclose."""
    assert np.allclose(
        custom_out.detach(),
        ref_out.detach(),
        atol=atol,
        rtol=rtol,
        equal_nan=True)


def assert_close_torch(
        custom_out: torch.Tensor,
        ref_out: torch.Tensor,
        atol: float = ATOL,
        rtol: float = RTOL) -> None:
    """Assert two tensors are close using torch.testing.assert_close."""
    torch.testing.assert_close(
        custom_out,
        ref_out,
        atol=atol,
        rtol=rtol,
        equal_nan=True)

# ── Reference Functions ─────────────────────────────────────────────────


def prelu_ref(x: torch.Tensor) -> torch.Tensor:
    """PReLU reference implementation with weight 0.25."""
    weight: torch.Tensor = torch.tensor([0.25])
    return F.prelu(x, weight)


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """Softmax reference implementation along last dimension."""
    return F.softmax(x, dim=-1)

# ── Test Cases ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("custom_cls, ref_fn", [
    (CustomReLU, F.relu),
    (CustomReLU6, F.relu6),
    (CustomPReLU, prelu_ref),
    (CustomSELU, F.selu),
    (CustomCELU, F.celu),
    (CustomGELU, F.gelu),
    (CustomSigmoid, F.sigmoid),
    (CustomMish, F.mish),
    (CustomSoftplus, F.softplus),
    (CustomTanh, F.tanh),
    (CustomSoftmax, softmax_ref),
    (CustomLeakyReLU, F.leaky_relu),
    (CustomELU, F.elu),
    (CustomSwish, F.silu),
])
class TestSimpleActivations:
    """Pytest parametrized tests that verify custom activation functions match PyTorch reference implementations."""

    def test_matches_pytorch(self,
                             custom_cls: nn.Module,
                             ref_fn: Callable[[torch.Tensor],
                                              torch.Tensor],
                             inputs_tensor: list[torch.Tensor]) -> None:
        """Verify custom activation output matches PyTorch reference for all parametrized inputs."""
        for input_tensor in inputs_tensor:
            assert_close_all(custom_cls()(input_tensor), ref_fn(input_tensor))
            assert_close_torch(
                custom_cls()(input_tensor),
                ref_fn(input_tensor))

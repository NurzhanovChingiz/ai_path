import torch
import pytest
from torch.nn import functional as F
from torch.testing import assert_close
import numpy as np

from activation.activation import (
    CustomReLU,
    CustomReLU6,
    CustomPReLU,
    CustomSELU,
    CustomCELU,
    CustomGELU,
    CustomSigmoid,
    CustomMish,
    CustomSoftplus,
    CustomTanh,
    CustomSoftmax,
    CustomLeakyReLU,
    CustomELU,
    CustomSwish,
)

ATOL = 1e-5 # 0.001%
RTOL = 1e-3 # 0.1%
# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def inputs_tensor():
    torch.manual_seed(42)
    
    inputs = [
        torch.randn(2, 3, 4),
        torch.randn(8, 16),
        torch.randn(2, 3, 4) * 100,
        torch.randn(1, 1),
        torch.randn(1, 1, 1),
        torch.randn(1, 1) *-1,
        torch.randn(1, 1, 1) *0,
        torch.randn(10, 10) *float('inf'),
        torch.randn(10, 10, 10) *float('-inf'),
        torch.randn(10, 10) *float('nan')
    ]
    return inputs
# ── Assertions ──────────────────────────────────────────────────────────────────
def assert_close_all(custom_out: torch.Tensor, ref_out: torch.Tensor, atol=ATOL, rtol=RTOL):
    assert np.allclose(custom_out.detach(), ref_out.detach(), atol=atol, rtol=rtol, equal_nan=True)
    
def assert_equal(custom_out: torch.Tensor, ref_out: torch.Tensor, atol=ATOL, rtol=RTOL):
    assert np.allclose(custom_out.detach(), ref_out.detach(), atol=atol, rtol=rtol, equal_nan=True)
    
def assert_close(custom_out: torch.Tensor, ref_out: torch.Tensor, atol=ATOL, rtol=RTOL):
    assert np.allclose(custom_out.detach(), ref_out.detach(), atol=atol, rtol=rtol, equal_nan=True)
# ── Reference Functions ──────────────────────────────────────────────────────────
def prelu_ref(x: torch.Tensor) -> torch.Tensor:
    weight = torch.tensor([0.25])
    return F.prelu(x, weight)
def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)

# ── Test Cases ──────────────────────────────────────────────────────────────────
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

    def test_matches_pytorch(self, custom_cls, ref_fn, inputs_tensor):
        for input_tensor in inputs_tensor:
            assert_close(custom_cls()(input_tensor), ref_fn(input_tensor))
            assert_close_all(custom_cls()(input_tensor), ref_fn(input_tensor))
            assert_equal(custom_cls()(input_tensor), ref_fn(input_tensor))
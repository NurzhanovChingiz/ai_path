import pytest
import torch
from typing import Any
from torch import nn
import numpy as np

from optimizers.learning.SGD import SGD
from optimizers.learning.SGD_with_momentum import SGD_with_momentum
from optimizers.learning.SGD_with_nesterov import SGD_with_nesterov
from optimizers.learning.SGD_with_weight_decay import SGD as SGD_weight_decay
from optimizers.learning.Adam import Adam
from optimizers.learning.AdamW import AdamW

ATOL: float = 1e-5  # 0.001%
RTOL: float = 1e-3  # 0.1%
SEED: int = 42


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_data(seed: int = SEED) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    return x, y


def run_optimizer(optimizer_cls: type,
                  model: nn.Module,
                  input_tensor: torch.Tensor,
                  output_tensor: torch.Tensor,
                  **optim_kwargs: Any) -> tuple[torch.Tensor,
                                                torch.Tensor]:
    optimizer = optimizer_cls(model.parameters(), **optim_kwargs)
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = nn.MSELoss()(output, output_tensor)
    loss.backward()
    optimizer.step()
    weight: torch.Tensor = model.weight.data  # type: ignore[assignment]
    bias: torch.Tensor = model.bias.data  # type: ignore[assignment]
    return weight, bias


def run_n_steps(optimizer_cls: type,
                model: nn.Module,
                x: torch.Tensor,
                y: torch.Tensor,
                n_steps: int,
                **optim_kwargs: Any) -> tuple[torch.Tensor,
                                              torch.Tensor]:
    optimizer = optimizer_cls(model.parameters(), **optim_kwargs)
    for _ in range(n_steps):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
    weight: torch.Tensor = model.weight.data  # type: ignore[assignment]
    bias: torch.Tensor = model.bias.data  # type: ignore[assignment]
    return weight, bias

# ── Assertions ──────────────────────────────────────────────────────────


def assert_close_all(
        custom_out: torch.Tensor,
        ref_out: torch.Tensor,
        atol: float = ATOL,
        rtol: float = RTOL) -> None:
    assert np.allclose(
        custom_out,
        ref_out,
        atol=atol,
        rtol=rtol,
        equal_nan=True)


def assert_close_torch(
        custom_out: torch.Tensor,
        ref_out: torch.Tensor,
        atol: float = ATOL,
        rtol: float = RTOL) -> None:
    torch.testing.assert_close(
        custom_out,
        ref_out,
        atol=atol,
        rtol=rtol,
        equal_nan=True)


# ── Test SGD ─────────────────────────────────────────────────────────────────
class TestSGD:

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_optimizer(
            SGD, model_custom, x, y, lr=0.01, inplace=inplace)
        w_ref, b_ref = run_optimizer(torch.optim.SGD, model_ref, x, y, lr=0.01)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_multi_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_n_steps(
            SGD, model_custom, x, y, n_steps=5, lr=0.01, inplace=inplace)
        w_ref, b_ref = run_n_steps(
            torch.optim.SGD, model_ref, x, y, n_steps=5, lr=0.01)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    def test_inplace_and_non_inplace_agree(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_ip = nn.Linear(1, 1)
        set_seed(SEED)
        model_nip = nn.Linear(1, 1)

        w_ip, b_ip = run_optimizer(SGD, model_ip, x, y, lr=0.01, inplace=True)
        w_nip, b_nip = run_optimizer(
            SGD, model_nip, x, y, lr=0.01, inplace=False)

        assert_close_torch(w_ip, w_nip)
        assert_close_torch(b_ip, b_nip)


# ── Test SGD with Weight Decay ───────────────────────────────────────────────
class TestSGDWeightDecay:

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_optimizer(
            SGD_weight_decay, model_custom, x, y, lr=0.01, inplace=inplace, weight_decay=0.01)
        w_ref, b_ref = run_optimizer(
            torch.optim.SGD, model_ref, x, y, lr=0.01, weight_decay=0.01)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_multi_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_n_steps(
            SGD_weight_decay, model_custom, x, y, n_steps=5, lr=0.01, inplace=inplace, weight_decay=0.01)
        w_ref, b_ref = run_n_steps(
            torch.optim.SGD, model_ref, x, y, n_steps=5, lr=0.01, weight_decay=0.01)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    def test_inplace_and_non_inplace_agree(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_ip = nn.Linear(1, 1)
        set_seed(SEED)
        model_nip = nn.Linear(1, 1)

        w_ip, b_ip = run_optimizer(
            SGD_weight_decay, model_ip, x, y, lr=0.01, inplace=True, weight_decay=0.01)
        w_nip, b_nip = run_optimizer(
            SGD_weight_decay, model_nip, x, y, lr=0.01, inplace=False, weight_decay=0.01)

        assert_close_torch(w_ip, w_nip)
        assert_close_torch(b_ip, b_nip)

    def test_zero_weight_decay_matches_plain_sgd(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_wd = nn.Linear(1, 1)
        set_seed(SEED)
        model_plain = nn.Linear(1, 1)

        w_wd, b_wd = run_optimizer(
            SGD_weight_decay, model_wd, x, y, lr=0.01, weight_decay=0)
        w_plain, b_plain = run_optimizer(SGD, model_plain, x, y, lr=0.01)

        assert_close_torch(w_wd, w_plain)
        assert_close_torch(b_wd, b_plain)


# ── Test SGD with Momentum ───────────────────────────────────────────────────
class TestSGDMomentum:

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_optimizer(
            SGD_with_momentum, model_custom, x, y, lr=0.01, inplace=inplace, momentum=0.9)
        w_ref, b_ref = run_optimizer(
            torch.optim.SGD, model_ref, x, y, lr=0.01, momentum=0.9)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_multi_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_n_steps(
            SGD_with_momentum, model_custom, x, y, n_steps=5, lr=0.01, inplace=inplace, momentum=0.9)
        w_ref, b_ref = run_n_steps(
            torch.optim.SGD, model_ref, x, y, n_steps=5, lr=0.01, momentum=0.9)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    def test_inplace_and_non_inplace_agree(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_ip = nn.Linear(1, 1)
        set_seed(SEED)
        model_nip = nn.Linear(1, 1)

        w_ip, b_ip = run_optimizer(
            SGD_with_momentum, model_ip, x, y, lr=0.01, inplace=True, momentum=0.9)
        w_nip, b_nip = run_optimizer(
            SGD_with_momentum, model_nip, x, y, lr=0.01, inplace=False, momentum=0.9)

        assert_close_torch(w_ip, w_nip)
        assert_close_torch(b_ip, b_nip)


# ── Test SGD with Nesterov ───────────────────────────────────────────────────
class TestSGDNesterov:

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_optimizer(
            SGD_with_nesterov, model_custom, x, y, lr=0.01, inplace=inplace, momentum=0.9, nesterov=True)
        w_ref, b_ref = run_optimizer(
            torch.optim.SGD, model_ref, x, y, lr=0.01, momentum=0.9, nesterov=True)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_multi_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_n_steps(
            SGD_with_nesterov, model_custom, x, y, n_steps=5, lr=0.01, inplace=inplace, momentum=0.9, nesterov=True)
        w_ref, b_ref = run_n_steps(
            torch.optim.SGD, model_ref, x, y, n_steps=5, lr=0.01, momentum=0.9, nesterov=True)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    def test_inplace_and_non_inplace_agree(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_ip = nn.Linear(1, 1)
        set_seed(SEED)
        model_nip = nn.Linear(1, 1)

        w_ip, b_ip = run_optimizer(
            SGD_with_nesterov, model_ip, x, y, lr=0.01, inplace=True, momentum=0.9, nesterov=True)
        w_nip, b_nip = run_optimizer(
            SGD_with_nesterov, model_nip, x, y, lr=0.01, inplace=False, momentum=0.9, nesterov=True)

        assert_close_torch(w_ip, w_nip)
        assert_close_torch(b_ip, b_nip)


# ── Test Adam ────────────────────────────────────────────────────────────────
class TestAdam:

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_optimizer(
            Adam, model_custom, x, y, lr=0.01, inplace=inplace, betas=(
                0.9, 0.999), eps=1e-8)
        w_ref, b_ref = run_optimizer(
            torch.optim.Adam, model_ref, x, y, lr=0.01, betas=(
                0.9, 0.999), eps=1e-8)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_multi_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_n_steps(
            Adam, model_custom, x, y, n_steps=5, lr=0.01, inplace=inplace, betas=(
                0.9, 0.999), eps=1e-8)
        w_ref, b_ref = run_n_steps(
            torch.optim.Adam, model_ref, x, y, n_steps=5, lr=0.01, betas=(
                0.9, 0.999), eps=1e-8)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    def test_inplace_and_non_inplace_agree(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_ip = nn.Linear(1, 1)
        set_seed(SEED)
        model_nip = nn.Linear(1, 1)

        w_ip, b_ip = run_optimizer(
            Adam, model_ip, x, y, lr=0.01, inplace=True, betas=(
                0.9, 0.999), eps=1e-8)
        w_nip, b_nip = run_optimizer(
            Adam, model_nip, x, y, lr=0.01, inplace=False, betas=(
                0.9, 0.999), eps=1e-8)

        assert_close_torch(w_ip, w_nip)
        assert_close_torch(b_ip, b_nip)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_custom_betas(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_optimizer(
            Adam, model_custom, x, y, lr=0.001, inplace=inplace, betas=(
                0.8, 0.99), eps=1e-10)
        w_ref, b_ref = run_optimizer(
            torch.optim.Adam, model_ref, x, y, lr=0.001, betas=(
                0.8, 0.99), eps=1e-10)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)


# ── Test AdamW ───────────────────────────────────────────────────────────────
class TestAdamW:

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_optimizer(
            AdamW, model_custom, x, y, lr=0.01, inplace=inplace, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.01)
        w_ref, b_ref = run_optimizer(
            torch.optim.AdamW, model_ref, x, y, lr=0.01, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.01)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_multi_step_matches_pytorch(self, inplace: bool) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_custom = nn.Linear(1, 1)
        set_seed(SEED)
        model_ref = nn.Linear(1, 1)

        w_custom, b_custom = run_n_steps(
            AdamW, model_custom, x, y, n_steps=5, lr=0.01, inplace=inplace, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.01)
        w_ref, b_ref = run_n_steps(
            torch.optim.AdamW, model_ref, x, y, n_steps=5, lr=0.01, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.01)

        assert_close_all(w_custom, w_ref)
        assert_close_torch(b_custom, b_ref)

    def test_inplace_and_non_inplace_agree(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_ip = nn.Linear(1, 1)
        set_seed(SEED)
        model_nip = nn.Linear(1, 1)

        w_ip, b_ip = run_optimizer(
            AdamW, model_ip, x, y, lr=0.01, inplace=True, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.01)
        w_nip, b_nip = run_optimizer(
            AdamW, model_nip, x, y, lr=0.01, inplace=False, betas=(
                0.9, 0.999), eps=1e-8, weight_decay=0.01)

        assert_close_torch(w_ip, w_nip)
        assert_close_torch(b_ip, b_nip)

    def test_zero_weight_decay_matches_adam(self) -> None:
        x, y = make_data()
        set_seed(SEED)
        model_adamw = nn.Linear(1, 1)
        set_seed(SEED)
        model_adam = nn.Linear(1, 1)

        w_adamw, b_adamw = run_optimizer(
            AdamW, model_adamw, x, y, lr=0.01, weight_decay=0, betas=(
                0.9, 0.999), eps=1e-8)
        w_adam, b_adam = run_optimizer(
            Adam, model_adam, x, y, lr=0.01, betas=(
                0.9, 0.999), eps=1e-8)

        assert_close_torch(w_adamw, w_adam)
        assert_close_torch(b_adamw, b_adam)

"""Muon optimizer implementation."""
import copy
import math
import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, ParamsT


def set_seed(seed: int = 42) -> None:
    """Set the seed for the random number generators.

    Args:
        seed: The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random Seed : {seed}")


def zero_power_via_newtonschulz5(
        grad: torch.Tensor,
        steps: int = 5,
        eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.

    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Args:
        grad: The gradient tensor.
        steps: The number of steps to run the iteration.
        eps: The epsilon value to prevent division by zero.

    Returns:
        The zero power of the gradient tensor.

    References:
        -Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
        -Suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if grad.ndim < 2:
        raise ValueError(f"grad must have at least 2 dimensions, got ndim={grad.ndim}")
    if steps > 100:
        raise ValueError(f"steps must be <= 100, got {steps}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: str | None,
               param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class Muon(Optimizer):
    """Muon is a variant of the Adam optimizer that uses orthogonalization to improve training stability.

    It is based on the idea that the gradient of the loss function is a linear combination of the parameters.
    By orthogonalizing the gradient, we can improve the training stability and convergence rate.

    References:
        - Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
        - Suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
            momentum: float = 0.95,
            nesterov: bool = True,
            steps: int = 5,
            eps: float = 1e-7,
            adjust_lr_fn: str | None = None) -> None:
        """Initialize the Muon optimizer.

        Args:
            params: The parameters to optimize.
            lr: The learning rate.
            weight_decay: The weight decay.
            momentum: The momentum.
            nesterov: Whether to use Nesterov momentum.
            steps: The number of steps to run the iteration.
            eps: The epsilon value to prevent division by zero.
            adjust_lr_fn: The learning rate adjustment function.
        """
        super().__init__(
            params,
              defaults={
                  "lr": lr,
                  "weight_decay": weight_decay,
                  "momentum": momentum,
                  "nesterov": nesterov,
                  "adjust_lr_fn": adjust_lr_fn,
                  "eps": eps,
                  "steps": steps,
              })
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.adjust_lr_fn = adjust_lr_fn
        self.eps = eps
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original", "match_rms_adamw"]:
            raise ValueError(
                'adjust_lr_fn must be None or "original" or "match_rms_adamw"'
            )
        if lr == 0 or lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if momentum == 0 or momentum <= 0 or momentum >= 1:
            raise ValueError(f"momentum must be in (0, 1), got {momentum}")
        if weight_decay == 0 or weight_decay <= 0:
            raise ValueError(f"weight_decay must be positive, got {weight_decay}")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None: # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: A closure that evaluates the loss.

        Returns:
            The loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            adjust_lr_fn = group["adjust_lr_fn"]
            eps = group["eps"]
            steps = group["steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p)
                v = state["v"]

                # Muon: momentum buffer, then orthogonalize the update (not
                # params)
                v.lerp_(grad, 1 - momentum)
                update = grad.lerp(v, momentum) if nesterov else v.clone()

                if p.ndim < 2:
                    p.mul_(1 - lr * weight_decay)
                    p.add_(update, alpha=-lr)
                    continue

                # 2D+ params: orthogonalize update via Newton-Schulz
                orig_shape = p.shape
                if p.ndim == 4:
                    update = update.view(update.size(0), -1)
                update = zero_power_via_newtonschulz5(
                    update, steps=steps, eps=eps)
                update = update.to(p.dtype)
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape)

                p.mul_(1 - lr * weight_decay)
                p.add_(update.reshape(orig_shape), alpha=-adjusted_lr)

        return loss


if __name__ == "__main__":
    seed = 42
    batch_size = 2
    in_features = 1
    out_features = 1
    lr = 1e-3
    weight_decay = 0.01
    momentum = 0.95
    nesterov = True
    steps = 5
    eps = 1e-7
    adjust_lr_fn = None  # or "original" or "match_rms_adamw"

    set_seed(seed)
    # Muon only supports 2D params
    model = nn.Linear(in_features, out_features, bias=False)
    custom_model = copy.deepcopy(model)
    custom_optimizer = Muon(
        custom_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
    )
    optimizer = torch.optim.Muon(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
    )
    custom_optimizer.zero_grad()
    optimizer.zero_grad()
    # Create dummy input and compute loss to generate gradients
    x = torch.randn(batch_size, in_features)
    y = torch.randn(batch_size, out_features)
    output = model(x)
    custom_output = custom_model(x)
    loss = nn.MSELoss()(output, y)
    custom_loss = nn.MSELoss()(custom_output, y)
    # Compute gradients
    loss.backward()
    custom_loss.backward()

    # Now we can call step()
    optimizer.step()
    custom_optimizer.step()
    print("custom_model.state_dict():", custom_model.state_dict())
    print("model.state_dict():", model.state_dict())
    print("custom_optimizer.state_dict():", custom_optimizer.state_dict())
    print("optimizer.state_dict():", optimizer.state_dict())
    print("custom_model.weight.data.clone():",
          custom_model.weight.data.clone())
    print("model.weight.data.clone():", model.weight.data.clone())
    # print(model.weight.data.clone())

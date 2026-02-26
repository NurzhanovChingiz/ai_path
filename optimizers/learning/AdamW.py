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


class AdamW(Optimizer):
    """AdamW optimizer.
    """
    def __init__(self,
                 params: ParamsT,
                 lr: float,
                 inplace: bool = True,
                 betas: tuple[float,
                              float] = (0.9,
                                        0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0) -> None:
        super().__init__(
            params,
            defaults=dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay))
        self.inplace = inplace

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
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                if weight_decay != 0:
                    if self.inplace:
                        # w = w * (1 - lr * weight_decay)
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        # w = w * (1 - lr * weight_decay)
                        p.data = p.data.clone() * (1 - lr * weight_decay)
                t = state["t"]
                m = state["m"]
                v = state["v"]
                if self.inplace:
                    # m = m * betas[0] + grad * (1 - betas[0])
                    m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    # v = v * betas[1] + grad * grad * (1 - betas[1])
                    v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                    # m_hat = m / (1 - betas[0] ** t)
                    m_hat = m / (1 - betas[0] ** t)
                    # v_hat = v / (1 - betas[1] ** t)
                    v_hat = v / (1 - betas[1] ** t)
                    # w = w - lr * m_hat / (v_hat.sqrt() + eps)
                    p.data.sub_(lr * m_hat / (v_hat.sqrt() + eps))
                else:
                    # m = m * betas[0] + grad * (1 - betas[0])
                    m = betas[0] * m + (1 - betas[0]) * grad
                    # v = v * betas[1] + grad * grad * (1 - betas[1])
                    v = betas[1] * v + (1 - betas[1]) * grad**2
                    # m_hat = m / (1 - betas[0] ** t)
                    m_hat = m / (1 - betas[0]**t)
                    # v_hat = v / (1 - betas[1] ** t)
                    v_hat = v / (1 - betas[1]**t)
                    # update = lr * m_hat / (v_hat.sqrt() + eps)
                    update = lr * m_hat / (v_hat.sqrt() + eps)

                    p.data = p.data.clone() - update  # w = w - update

                    state["m"] = m.clone()
                    state["v"] = v.clone()
                    state["t"] = t
        return loss


# testing
if __name__ == "__main__":
    set_seed(42)
    model = nn.Linear(1, 1)
    optimizer = AdamW(
        model.parameters(),
        lr=0.01,
        inplace=False,
        betas=(
            0.9,
            0.999),
        eps=1e-10,
        weight_decay=0.01)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-10, weight_decay=0.01)
    optimizer.zero_grad()
    # Create dummy input and compute loss to generate gradients
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    output = model(x)
    loss = nn.MSELoss()(output, y)

    # Compute gradients
    loss.backward()

    # Now we can call step()
    optimizer.step()
    print(optimizer.state_dict())
    print(model.state_dict())
    print(model.weight.data.clone())

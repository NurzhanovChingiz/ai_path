"""SGD optimizer with momentum implementation."""
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


class SGD_with_momentum(Optimizer):
    """SGD optimizer with momentum."""
    def __init__(
            self,
            params: ParamsT,
            lr: float,
            inplace: bool = True,
            momentum: float = 0.9) -> None:
        """Initialize the SGD optimizer with momentum.

        Args:
            params: The parameters to optimize.
            lr: The learning rate.
            inplace: Whether to use inplace operations.
            momentum: The momentum.
        """
        super().__init__(params, defaults=dict(lr=lr, momentum=momentum))
        self.momentum = momentum
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
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p)

                v = state["v"]
                if self.inplace:
                    v.mul_(momentum).add_(grad)  # v = v * momentum + grad
                    p.data.sub_(lr * v)  # w = w - lr * v
                else:
                    v = momentum * v + grad  # v = momentum * v + grad
                    update = lr * v  # update = lr * v
                    p.data = p.data.clone() - update  # w = w - update
                    state["v"] = v.clone()
        return loss


# testing
if __name__ == "__main__":
    set_seed(42)
    model = nn.Linear(1, 1)
    optimizer = SGD_with_momentum(
        model.parameters(),
        lr=0.01,
        inplace=False,
        momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

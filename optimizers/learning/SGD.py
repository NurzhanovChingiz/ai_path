"""SGD optimizer implementation."""
import random
from collections.abc import Callable

import numpy as np
import torch
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


class SGD(Optimizer):
    """SGD optimizer."""
    def __init__(
            self,
            params: ParamsT,
            lr: float,
            inplace: bool = True) -> None:
        """Initialize the SGD optimizer.

        Args:
            params: The parameters to optimize.
            lr: The learning rate.
            inplace: Whether to use inplace operations.
        """
        super().__init__(params, defaults={"lr": lr})
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.inplace:
                    p.data.sub_(p.grad * lr)  # w = w - grad * lr
                else:
                    update = p.grad * lr  # update = grad * lr
                    p.data = p.data.clone() - update  # w = w - update
        return loss

import random
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer, ParamsT


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random Seed : {seed}")


class SGD(Optimizer):
    def __init__(
            self,
            params: ParamsT,
            lr: float,
            inplace: bool = True) -> None:
        super().__init__(params, defaults=dict(lr=lr))
        self.inplace = inplace

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None: # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.inplace:
                    p.data.sub_(p.grad * lr)  # w = w - grad * lr
                else:
                    update = p.grad * lr  # update = grad * lr
                    p.data = p.data.clone() - update  # w = w - update
        return loss


# testing
if __name__ == "__main__":
    set_seed(42)
    model = nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.01, inplace=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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

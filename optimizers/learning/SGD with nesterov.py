from torch.optim import Optimizer
from typing import Any
from torch import nn
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random Seed : {seed}")

class SGD_with_nesterov(Optimizer):
    def __init__(self, params, lr, inplace=True, momentum=0.9, nesterov=True) -> None:
        super().__init__(params, defaults=dict(lr=lr, momentum=momentum, nesterov=nesterov))
        self.momentum=momentum
        self.inplace=inplace
        self.nesterov=nesterov
        
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p)

                v = state["v"]
                if self.inplace:
                    v.mul_(momentum).add_(grad) # v = v * momentum + grad
                    state['v'] = v.clone()
                    if nesterov:
                        v.mul_(momentum).add_(grad) # v = v * momentum + grad
                    p.data.sub_(lr * v) # w = w - lr * v
                else:
                    v = momentum*v + grad # v = momentum * v + grad
                    state['v'] = v.clone()
                    if nesterov:
                        v = momentum * v + grad # v = momentum * v + grad
                    update = lr * v # update = lr * v
                    p.data = p.data.clone() - update
                
# testing
if __name__ == "__main__":
    set_seed(42)
    model = nn.Linear(1, 1)
    optimizer = SGD_with_nesterov(model.parameters(), lr=0.01, inplace=True, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
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
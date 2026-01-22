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

class SGD(Optimizer):
    def __init__(self, params, lr, inplace=True) -> None:
        super().__init__(params, defaults=dict(lr=lr))
        self.inplace=inplace
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if self.inplace:
                    p.data.sub_(p.grad*lr)
                else:
                    update = p.grad*lr
                    p.data = p.data + update 
# testing
if __name__ == "__main__":
    set_seed(42)
    model = nn.Linear(10, 10)
    optimizer = SGD(model.parameters(), lr=0.01, inplace=True)
    optimizer.zero_grad()
    # Create dummy input and compute loss to generate gradients
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    output = model(x)
    loss = nn.MSELoss()(output, y)
    
    # Compute gradients
    loss.backward()
    
    # Now we can call step()
    optimizer.step()
    print(optimizer.state_dict())
    print(model.weight.data.clone())
    
    
    
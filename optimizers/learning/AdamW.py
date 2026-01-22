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

class AdamW(Optimizer):
    def __init__(self, params, lr, inplace=True, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) -> None:
        super().__init__(params, defaults=dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.inplace=inplace
        
    @torch.no_grad()
    def step(self):
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
                        p.data.mul_(1 - lr * weight_decay) # p.data = p.data * (1 - lr * weight_decay)
                    else:
                        p.data = p.data.clone() * (1 - lr * weight_decay) # p.data = p.data * (1 - lr * weight_decay)
                t = state["t"]
                m = state["m"]
                v = state["v"]
                if self.inplace:
                    m.mul_(betas[0]).add_(grad, alpha=1 - betas[0]) # m = m * betas[0] + grad * (1 - betas[0])
                    v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1]) # v = v * betas[1] + grad * grad * (1 - betas[1])
                    m_hat = m / (1 - betas[0] ** t) # m_hat = m / (1 - betas[0] ** t)
                    v_hat = v / (1 - betas[1] ** t) # v_hat = v / (1 - betas[1] ** t)
                    p.data.sub_(lr * m_hat / (v_hat.sqrt() + eps)) # p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps)
                else:
                    m = betas[0]*m + (1-betas[0])*grad # m = m * betas[0] + grad * (1 - betas[0])
                    v = betas[1]*v + (1-betas[1])*grad**2 # v = v * betas[1] + grad * grad * (1 - betas[1])
                    m_hat = m/(1-betas[0]**t) # m_hat = m / (1 - betas[0] ** t)
                    v_hat = v/(1-betas[1]**t) # v_hat = v / (1 - betas[1] ** t)
                    update = lr * m_hat / (v_hat.sqrt() + eps) # update = lr * m_hat / (v_hat.sqrt() + eps)

                    p.data = p.data.clone() - update # p.data = p.data - update

                    state["m"] = m.clone()
                    state["v"] = v.clone()
                    state["t"] = t

        
# testing
if __name__ == "__main__":
    set_seed(42)
    model = nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=0.01, inplace=False, betas=(0.9, 0.999), eps=1e-10, weight_decay=0.01)
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
    
    
    
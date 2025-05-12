import torch 
import torch.nn as nn
from torch.nn.modules import Module

class CustomRuLU(Module):
    """
    Custom ReLU activation function.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.where(x > 0, x, torch.zeros_like(x)), x)
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
class CustomSigmoid(Module):
    """
    Custom Sigmoid activation function.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
if __name__ == "__main__":
    x = torch.tensor([[0, 0, 7],
        [-1, 0, 1],
        [3, 0, 5]])
    # print(x)
    # relu = CustomRuLU()
    # print(relu(x))
    print(CustomRuLU()(x))
    print(CustomSigmoid()(x))


import torch 
from torch import nn
from torch.nn import functional as F

class CustomReLU(nn.Module):
    """
    Custom ReLU activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, min=0)
    
class CustomReLU6(nn.Module):
    """
    Custom ReLU6 activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, min=0, max=6)

class CustomPReLU(nn.Module):
    """
    Custom PReLU activation function.
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_parameters).fill_(init))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if x.dim() > 1 and self.weight.numel() > 1:
            # Reshape weight to (1, num_parameters, 1, ..., 1) for broadcasting
            shape = [1] * x.dim()
            shape[1] = self.weight.numel()
            weight = self.weight.reshape(shape)
        return torch.where(x >= 0, x, weight * x)
    

class CustomSELU(nn.Module):
    """
    Custom SELU activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        rhs = alpha * torch.expm1(x)

        return scale * torch.where(x > 0, x, rhs)
    
class CustomCELU(nn.Module):
    """
    Custom CELU activation function.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * (torch.exp(x / self.alpha) - 1))
class CustomGELU(nn.Module):
    """
    Custom GELU activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(0.797884561 * (x + 0.044715 * x**3)))
class CustomSigmoid(nn.Module):
    """
    Custom Sigmoid activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))

class CustomMish(nn.Module):
    """
    Custom Mish activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class CustomSoftplus(nn.Module):
    """
    Custom Softplus activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.exp(x))

class CustomTanh(nn.Module):   
    """
    Custom Tanh activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0

class CustomSoftmax(nn.Module):
    """
    Custom Softmax activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)   
class CustomLeakyReLU(nn.Module):
    """
    Custom Leaky ReLU activation function.
    """
    def __init__(self, negative_slope: float = 0.01):
        super(CustomLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, min=0) + self.negative_slope * torch.clip(x, max=0)

class CustomELU(nn.Module):
    """
    Custom ELU activation function.
    """
    def __init__(self, alpha: float = 1.0):
        super(CustomELU, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
    
class CustomSwish(nn.Module):
    """
    Custom Swish activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    


    

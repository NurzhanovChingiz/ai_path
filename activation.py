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
from torch import nn

class CustomPReLU(nn.Module):
    """
    Custom PReLU activation function.
    """
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_parameters).fill_(init))

    def forward(self, x):
        return torch.where(x >= 0, x, self.weight * x)

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
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

class CustomSoftmax(nn.Module):
    """
    Custom Softmax activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)   
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
    
class CustomSwiff(nn.Module):
    """
    Custom Swish activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
if __name__ == "__main__":
    x = torch.tensor([[0, 0, 7],
        [-1, 0, 1],
        [3, -5, 5]])
    from torch import nn
    nn.LeakyReLU()
    print('CustomReLU', CustomReLU()(x))
    print('CustomReLU6', CustomReLU6()(x))
    print('CustomPReLU', CustomPReLU()(x))
    print('CustomSELU', CustomSELU()(x))
    print('CustomCELU', CustomCELU()(x))
    print('CustomGELU', CustomGELU()(x))
    print('CustomSigmoid', CustomSigmoid()(x))
    print('CustomSoftplus', CustomSoftplus()(x))
    print('CustomTanh', CustomTanh()(x))
    print('CustomSoftmax', CustomSoftmax()(x))
    print('CustomLeakyReLU', CustomLeakyReLU()(x))
    print('CustomELU', CustomELU()(x))
    print('CustomSwiff', CustomSwiff()(x))


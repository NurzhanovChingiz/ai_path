"""Custom activation functions for neural networks."""

import torch
from torch import nn
from torch.nn import functional as F


class CustomReLU(nn.Module):
    """Custom ReLU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU activation (max(0, x))."""
        return torch.clip(x, min=0)


class CustomReLU6(nn.Module):
    """Custom ReLU6 activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU6 activation (clamp to [0, 6])."""
        return torch.clip(x, min=0, max=6)


class CustomPReLU(nn.Module):
    """Custom PReLU activation function."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        """Initialize PReLU with learnable per-channel parameters."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PReLU activation with learnable negative slope."""
        weight: torch.Tensor = self.weight
        if x.dim() > 1 and self.weight.numel() > 1:
            shape = [1] * x.dim()
            shape[1] = self.weight.numel()
            weight = self.weight.reshape(shape)
        return torch.where(x >= 0, x, weight * x)


class CustomSELU(nn.Module):
    """Custom SELU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SELU activation (Scaled Exponential Linear Unit)."""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        rhs = alpha * torch.expm1(x)

        return scale * torch.where(x > 0, x, rhs)


class CustomCELU(nn.Module):
    """Custom CELU activation function."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize CELU with alpha parameter for negative values."""
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CELU activation (Continuously Differentiable ELU)."""
        return torch.where(x > 0, x, self.alpha *
                           (torch.exp(x / self.alpha) - 1))


class CustomGELU(nn.Module):
    """Custom GELU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation (Gaussian Error Linear Unit)."""
        inner = x / 1.4142135623730951
        result = 0.5 * x * (1.0 + torch.erf(inner))
        return torch.where(torch.isfinite(x), result, torch.nan)


class CustomSigmoid(nn.Module):
    """Custom Sigmoid activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid activation 1/(1 + exp(-x))."""
        return 1 / (1 + torch.exp(-x))


class CustomMish(nn.Module):
    """Custom Mish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mish activation x * tanh(softplus(x))."""
        return x * torch.tanh(F.softplus(x))


class CustomSoftplus(nn.Module):
    """Custom Softplus activation function."""

    def forward(self, x: torch.Tensor, beta: float = 1.0,
                threshold: float = 20.0) -> torch.Tensor:
        """Apply Softplus activation ln(1 + exp(x)) with optional beta and threshold."""
        scaled = beta * x
        return torch.where(
            scaled > threshold,
            x,
            torch.log(1 + torch.exp(scaled)) / beta,
        )


class CustomTanh(nn.Module):
    """Custom Tanh activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tanh activation."""
        return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0


class CustomSoftmax(nn.Module):
    """Custom Softmax activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax along the last dimension."""
        x_max = torch.max(x, dim=-1, keepdim=True).values
        e_x = torch.exp(x - x_max)
        return e_x / torch.sum(e_x, dim=-1, keepdim=True)


class CustomLeakyReLU(nn.Module):
    """Custom Leaky ReLU activation function."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        """Initialize LeakyReLU with negative slope for x < 0."""
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LeakyReLU with learned negative slope for x < 0."""
        return torch.clip(x, min=0) + self.negative_slope * \
            torch.clip(x, max=0)


class CustomELU(nn.Module):
    """Custom ELU activation function."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize ELU with alpha parameter for negative values."""
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ELU activation (Exponential Linear Unit)."""
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))


class CustomSwish(nn.Module):
    """Custom Swish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation x * sigmoid(x)."""
        return x * torch.sigmoid(x)

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
class CustomGELU(nn.Module):
    """
    Custom GELU activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(0.797884561 * (x + 0.044715 * x**3)))
if __name__ == "__main__":
    # Example usage
    # Plot the activation functions
    import matplotlib.pyplot as plt
    import numpy as np
    x = torch.linspace(-5, 5, 100)
    plt.figure(figsize=(10, 6))
    plt.title("Custom PReLU Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid()
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.plot(x.numpy(), CustomGELU()(x).detach().numpy(), label='Custom PReLU', color='blue')
    plt.legend()
    plt.show()
    # Plot the activation functions
    plt.figure(figsize=(10, 6))
    plt.title("PReLU Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid()
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.plot(x.numpy(), nn.GELU()(x).detach().numpy(), label='Custom PReLU', color='blue')
    plt.legend()
    plt.show()
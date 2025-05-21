import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from activation import *
class CustomMish(nn.Module):
    """
    Custom Mish activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def plot_activation_function(activation_func, x_range=(-5, 5), num_points=1000):
    """
    Plot the activation function.
    
    Parameters:
    - activation_func: The activation function to plot.
    - x_range: The range of x values to plot.
    - num_points: The number of points to plot.
    """
    print(f"Plotting {activation_func.__name__} activation function")

    x = torch.linspace(-1, 1, num_points)
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), activation_func()(x).detach().numpy(), color='blue')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(f"Activation Function: {activation_func.__name__}")
    plt.grid()
    plt.savefig(f"results/activation_fun/{activation_func.__name__}.png")
    plt.show()
if __name__ == "__main__":
    # Example usage
    # Plot the activation functions
    activatetion_functions_custom = [
        CustomReLU,
        CustomReLU6,
        CustomPReLU,
        CustomSELU,
        CustomCELU,
        CustomGELU,
        CustomSigmoid,
        CustomMish,
        CustomSoftplus,
        CustomTanh,
        CustomSoftmax,
        CustomLeakyReLU,
        CustomELU,
        CustomSwiff
    ]
    activatetion_functions_original = [
        nn.ReLU,
        nn.ReLU6,
        nn.PReLU,
        nn.SELU,
        nn.CELU,
        nn.GELU,
        nn.Sigmoid,
        nn.Mish,
        nn.Softplus,
        nn.Tanh,
        nn.Softmax,
        nn.LeakyReLU,
        nn.ELU
    ]
        
        
    for func in zip(activatetion_functions_custom, activatetion_functions_original):
        custom_func, original_func = func
        plot_activation_function(custom_func)
        plot_activation_function(original_func)


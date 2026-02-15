import torch
from torch import nn
from torch.nn import functional as F
from activation import *
import matplotlib.pyplot as plt
import os
from config import CFG

def plot_activation_function(activation_func: type[nn.Module], x_range: tuple[int, int] = (-5, 5), num_points: int = 1000, save: bool = True) -> None:
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
    if save:
        save_dir = os.getcwd() + CFG.PATH_TO_SAVE
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{activation_func.__name__}.png")
        plt.savefig(save_path)
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
        plot_activation_function(custom_func, save=CFG.IS_SAVE)
        plot_activation_function(original_func, save=CFG.IS_SAVE)


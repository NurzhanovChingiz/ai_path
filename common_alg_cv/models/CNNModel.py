"""CNN model implementation."""

from typing import cast

import torch
from torch import nn


class CNNModel(nn.Module):
    """Simple convolutional neural network for image classification."""
    def __init__(self) -> None:
        """Initialize the CNN model layers."""
        super().__init__()

        # Convolution 1
        self.relu1 = nn.ReLU()
        self.cnn1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0)

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.relu2 = nn.ReLU()
        self.cnn2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0)

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv layers and classifier."""
        # Convolution 1
        x = self.relu1(self.cnn1(x))
        # Max pool 1
        x = self.maxpool1(x)

        # Convolution 2
        x = self.relu2(self.cnn2(x))

        # Max pool 2
        x = self.maxpool2(x)

        # flatten
        x = x.flatten(1)

        # Linear function (readout)
        return cast("torch.Tensor", self.fc1(x))

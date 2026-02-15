from torch import Tensor
from torch import nn
from typing import Callable, Optional


class MobileNetV1(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False) # 224x224x3 -> 112x112x32
        self.bn1 = nn.BatchNorm2d(32) # 112x112x32
        self.conv_dw1 = DepthWiseConv2d(32, 64, 1) # 112x112x32 -> 112x112x64
        self.conv_dw2 = DepthWiseConv2d(64, 128, 2) # 112x112x64 -> 56x56x128
        self.conv_dw3 = DepthWiseConv2d(128, 128, 1) # 56x56x128 -> 56x56x128
        self.conv_dw4 = DepthWiseConv2d(128, 256, 2) # 56x56x128 -> 28x28x256
        self.conv_dw5 = DepthWiseConv2d(256, 256, 1) # 28x28x256 -> 28x28x256
        self.conv_dw6 = DepthWiseConv2d(256, 512, 2) # 28x28x256 -> 14x14x512
        self.conv_dw7 = DepthWiseConv2d(512, 512, 1) # 14x14x512 -> 14x14x512
        self.conv_dw8 = DepthWiseConv2d(512, 512, 1) # 14x14x512 -> 14x14x512
        self.conv_dw9 = DepthWiseConv2d(512, 512, 1) # 14x14x512 -> 14x14x512
        self.conv_dw10 = DepthWiseConv2d(512, 512, 1) # 14x14x512 -> 14x14x512
        self.conv_dw11 = DepthWiseConv2d(512, 512, 1) # 14x14x512 -> 14x14x512
        self.conv_dw12 = DepthWiseConv2d(512, 1024, 2)  # 14x14x512 -> 7x7x1024
        self.conv_dw13 = DepthWiseConv2d(1024, 1024, 1) # 7x7x1024 -> 7x7x1024
        # Global Average Pooling
        # self.avgpool = nn.AdaptiveAvgPool2d(1) # 7x7x1024 -> 1x1x1024
        self.avgpool = nn.AvgPool2d(7, stride=1)  # 7x7x1024 -> 1x1x1024
        self.classifier = nn.Linear(1024, num_classes)
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=False)(out)

        out = self.conv_dw1(out)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = self.conv_dw10(out)
        out = self.conv_dw11(out)
        out = self.conv_dw12(out)
        out = self.conv_dw13(out)

        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.classifier(out)

        return Tensor(out)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


class DepthWiseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride must be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv(x)

        return out
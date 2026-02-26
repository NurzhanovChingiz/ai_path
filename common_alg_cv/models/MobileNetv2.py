from collections.abc import Callable

from torch import Tensor, nn
from torchvision.ops.misc import Conv2dNormActivation  # type: ignore[import-untyped]
    

__all__ = ['MobileNetV2',  "InvertedResidual"]

def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    Args:
        v: original value
        divisor: the divisor to make v divisible by
        min_value: minimum value to return
    Returns:
        new_v: adjusted value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """Inverted Residual Block.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for depthwise convolution
        expand_ratio: Expansion ratio for the block
        norm_layer: Normalization layer to use
    Returns:
        Output tensor after applying the block
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: int,
            norm_layer: Callable[..., nn.Module] | None = None
    ) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_channels = int(round(in_channels * expand_ratio))

        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        layers: list[nn.Module] = []

        if expand_ratio != 1:
            # point-wise
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                    inplace=True,
                    bias=False,
                    )
            )
            
        # Depth-wise + point-wise layer
        layers.extend(
            [
                # Depth-wise
                Conv2dNormActivation(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_channels,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                    inplace=True,
                    bias=False,
                ),
                # point-wise layer
                nn.Conv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                    ),
                norm_layer(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)  # type: ignore[no-any-return]
        else:
            return self.conv(x)  # type: ignore[no-any-return]

class MobileNetV2(nn.Module):
    """MobileNet V2 main class

    Args:
        num_classes (int): Number of classes
        width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
        inverted_residual_setting: Network structure
        round_nearest (int): Round the number of channels in each layer to be a multiple of this number
        Set to 1 to turn off rounding
        block: Module specifying inverted residual building block for mobilenet
        norm_layer: Module specifying the normalization layer to use
        dropout (float): The dropout probability
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Returns:
        MobileNetV2 model
    """
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: list[list[int]] | None = None,
        round_nearest: int = 8,
        block: Callable[..., nn.Module] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
        dropout: float = 0.2,
        pretrained: bool = False,
            ) -> None:
        super().__init__()
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        input_channel = 32
        last_channel = 1280
        
        mobilenet_v2_inverted_residual_cfg: list[list[int]] = [
            # expand_ratio, out_channels, repeated times, stride
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        if inverted_residual_setting is None:
            inverted_residual_setting = mobilenet_v2_inverted_residual_cfg
            
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        classifier_channels = int(self.last_channel * max(1.0, width_mult))

        # building first layer
        features: list[nn.Module] = [
            Conv2dNormActivation(3,
                                 input_channel,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=nn.ReLU6,
                                 inplace=True,
                                 bias=False,
                                 )
        ]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel        
                
        # building last several layers       
        features.append(
            Conv2dNormActivation(input_channel,
                                 classifier_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm_layer=norm_layer,
                                 activation_layer=nn.ReLU6,
                                 inplace=True,
                                 bias=False,
                                 ),
        )
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_channels, num_classes),
        )

        # Initialize neural network weights
        self._initialize_weights(pretrained, num_classes)
        
                
    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out  # type: ignore[no-any-return]
    
    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)
        return out  # type: ignore[no-any-return]
    
    def _initialize_weights(self, pretrained: bool, num_classes: int) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not pretrained:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and not pretrained:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and not pretrained:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if pretrained:
            self._pretrained()
            # if self.num_classes != 1000, we keep classifier randomly initialized
            if num_classes != 1000:
                in_features = self.classifier[-1].in_features  # type: ignore[union-attr]
                self.classifier[-1] = nn.Linear(in_features, num_classes)  # type: ignore[arg-type]
                
    def _pretrained(self) -> None:
        """Load pretrained weights from torchvision"""
        from torchvision.models import (  # type: ignore[import-untyped]
            MobileNet_V2_Weights, mobilenet_v2)

        pretrained_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        state_dict = pretrained_model.state_dict()

        # For running classifier when num_classes != 1000
        self.load_state_dict(state_dict, strict=False)
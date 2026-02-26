import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        pretrained: bool = False,
        ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.init_weights(pretrained, num_classes)
    # Support torch.script function
    def forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_impl(x)
        return x

    def init_weights(self, pretrained: bool, num_classes: int) -> None:
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
        state_dict = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
            progress=True,
        )
        self.load_state_dict(state_dict)
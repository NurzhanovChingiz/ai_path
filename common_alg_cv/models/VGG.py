from torch import nn
import torch

cfg = {
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG21': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 'M'],
    'VGG23': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features: list, num_classes: int = 1000, p: float = 0.5, init_weights: bool = True) -> None:
        super().__init__()
        self.features = self._make_layers(features)
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc1', nn.Linear(512 * 1 * 1, 4096)) # adjust for input size, original was 7*7
        self.classifier.add_module('relu1', nn.ReLU(inplace=True))
        
        

        self.classifier.add_module('dropout1', nn.Dropout(p=p))
        self.classifier.add_module('fc2', nn.Linear(4096, 4096))
        self.classifier.add_module('relu2', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout2', nn.Dropout(p=p))
        self.classifier.add_module('fc3', nn.Linear(4096, num_classes))

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if device.type != 'cuda':
            self.classifier.dropout1 = nn.Identity()
            self.classifier.dropout2 = nn.Identity()
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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


    def _make_layers(self, cfg: list, batch_norm: bool = False) -> nn.Sequential:
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
from torch import nn
import torch
import torch.nn.functional as F

cfg = {
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG21': [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 'M'],
    'VGG23': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 512, 'M']
}
def _make_layers(self, cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x =='M':
            layers += [nn.MaxPool2d()]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1),
                      nn.BatchNorm2d(),
                      F.relu(),
                      F.elu()]
    layers += [nn.AdaptiveAvgPool2d()]
    return nn.Sequential(*layers)
        
        
class VGG(nn.Module):
    
    def __init__(self, vgg_name):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self._initialize_weight()
        
    def forward(self, x):
        img_input = x.view(-1, 1, 28, 28)
        # image too small so upscale it to -1,1,56,56
#         img_input = nn.functional.interpolate(img_input, scale_factor=2, mode='bilinear')
#         print(img_input.shape)
        img_input = (img_input - torch.mean(img_input))/torch.std(img_input)
        
        out = self.features(img_input)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
    def _make_layers(self, cfg):
        layers = [] # List to hold the layers
        in_channels = 1 # Initial number of input channels
        for x in cfg: # Iterate through the configuration
            if x == 'M': # If the element is 'M', add a Max Pooling layer
                layers += [nn.MaxPool2d(kernel_size=1, stride=1)] # Default kernel size and stride added for Max Pooling
            else: # Otherwise, create a convolutional block
                layers += [
                    nn.Conv2d(in_channels, x, padding=1, stride=1), # Convolutional layer with specified output channels
                    nn.BatchNorm2d(x), # Batch Normalization layer
                    F.relu(), # ReLU activation function
                    F.elu()   # ELU activation function
                ]
                in_channels = x # Update the number of input channels for the next iteration
        layers += [nn.AdaptiveAvgPool2d((3,3))] # Add Adaptive Average Pooling layer with output size (3,3)
        return nn.Sequential(*layers) # Return the layers as a sequential model

    
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
import torch_directml
from torch import nn
import torch
from models.ResNet import ResNet, cfg_resnet, BasicBlock
from models.MobileNetv1 import MobileNetV1
from alg_cv.set_seed import set_seed
from alg_cv.clear_gpu import clear_memory
class CFG:
    SEED = 42
    set_seed(SEED)
    clear_memory()
    
    SHOW_IMG = False
    IMG_SIZE = 28
    BATCH_SIZE = 64
    EPOCHS = 2
    LR = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    NUM_CLASSES = 10
    
    # Define the model, loss function, and optimizer
    DEVICE = torch_directml.device(torch_directml.default_device())
    # MODEL = ResNet(BasicBlock, cfg_resnet['ResNet34']).to(DEVICE)
    MODEL = MobileNetV1(num_classes=NUM_CLASSES).to(DEVICE)
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

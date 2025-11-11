import torch_directml
from torch import nn
import torch
from models.ResNet import ResNet, cfg_resnet, BasicBlock
from models.MobileNetv1 import MobileNetV1
from alg_cv.set_seed import set_seed
from alg_cv.clear_gpu import clear_memory
from alg_cv.summary import summary
from torch.backends import cudnn


class CFG: 
    SEED: int = 42
    set_seed(SEED)
    clear_memory()
    cudnn.benchmark = True

    SHOW_IMG: bool = False
    IMG_SIZE: int = 28
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    MOMENTUM: float = 0.9
    NUM_CLASSES: int = 10

    # Define the model, loss function, and optimizer
    DEVICE = torch_directml.device(torch_directml.default_device())
    # MODEL = ResNet(BasicBlock, cfg_resnet['ResNet34']).to(DEVICE)
    MODEL = MobileNetV1(num_classes=NUM_CLASSES).to(DEVICE)
    summary(MODEL)
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

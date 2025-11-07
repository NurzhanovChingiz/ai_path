import torch_directml
class CFG:
    SEED = 42
    SHOW_IMG = False
    IMG_SIZE = 28
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    NUM_CLASSES = 10
    DEVICE = torch_directml.device(torch_directml.default_device())
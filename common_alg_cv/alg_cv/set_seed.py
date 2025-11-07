import torch
import numpy as np
import random

def set_seed(SEED: int = 42):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print('Random Seed : {0}'.format(SEED))
    
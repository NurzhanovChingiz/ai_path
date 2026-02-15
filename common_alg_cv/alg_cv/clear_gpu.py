import torch
import time
import gc

def clear_memory(verbose: bool = True) -> None:
    stt = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    gc.collect()

    if verbose:
        print('Cleared memory.  Time taken was %f secs' % (time.time() - stt))
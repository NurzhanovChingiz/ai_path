# RLE encoding for img
import numpy as np
import json

def rle_encode(mask):
    """RLE encoding"""
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    mask = (mask > 0).astype(np.uint8)
    
    if mask.sum() == 0:
        return json.dumps([])
    
    pixels = mask.T.flatten()
    runs = []
    prev = 0
    pos = 0
    
    for i, pixel in enumerate(pixels):
        if pixel != prev:
            if prev == 1:
                runs.extend([pos + 1, i - pos])
            if pixel == 1:
                pos = i
            prev = pixel
    
    if prev == 1:
        runs.extend([pos + 1, len(pixels) - pos])
    
    return json.dumps([int(x) for x in runs])
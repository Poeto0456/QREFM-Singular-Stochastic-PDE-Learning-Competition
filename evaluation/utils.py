import numpy as np
import torch

def to_numpy(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()
import numpy as np

_MIN_CLIP = -7
_MAX_CLIP = 7


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    np.clip(x, _MIN_CLIP, _MAX_CLIP)
    return np.exp(x) / (1+np.exp(x))

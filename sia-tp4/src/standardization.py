import scipy as sp
import numpy as np
from numpy._typing import NDArray


def feature_scaling(x: NDArray, a: float, b: float):
    return a + (x - np.min(x)) * (b - a) / (np.max(x) - np.min(x))


def z_score(x: NDArray):
    return sp.stats.zscore(x)


def unit_vector(x: NDArray):
    return x / np.linalg.norm(x)

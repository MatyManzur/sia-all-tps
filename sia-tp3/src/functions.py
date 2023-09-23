from typing import Callable
import numpy as np
from numpy._typing import NDArray

Activation_Function = Callable[[NDArray], NDArray]

BETA = 1


##LAS FUNCIONES DEBERIAN RECIBIR UN VECTOR COLUMNA CON LA SUMA DE LOS Wi*Xi

def identity(x: NDArray | float) -> NDArray:
    return x

def derivative_identity(x: NDArray | float) -> NDArray:
    return np.apply_along_axis(lambda h: h-h+1, 0, x)  # no preguntes


def identity_normalization(x: NDArray | float) -> NDArray | float:
    return x


def sign(x: NDArray | float) -> NDArray | float:
    return np.apply_along_axis(inclusive_sign, 0, x)


def inclusive_sign(x) -> int:
    return -1 + 2*(x >= 0)


def hiperbolic(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return BETA * np.tanh(x)
    return np.apply_along_axis(lambda h: BETA * np.tanh(h), 0, x)


def hiperbolic_derivative(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return BETA * (1 - (BETA * np.tanh(x)) ** 2)
    return np.apply_along_axis(lambda h: BETA * (1 - (BETA * np.tanh(h)) ** 2), 0, x)


def hiperbolic_normalization(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return x / abs(x)
    max_value = max(abs(x))
    return np.apply_along_axis(lambda h: h/max_value, 0, x)


def sigmoid(x: NDArray) -> NDArray:
    if isinstance(x, float) or isinstance(x, int):
        return 1 / (1 + np.exp(-2 * BETA * x))
    return np.apply_along_axis(lambda h: 1 / (1 + np.exp(-2 * BETA * h)), 0, x)


def sigmoid_derivative(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return 2 * BETA * (1 / (1 + np.exp(-2 * BETA * x))) * (1 - (1 / (1 + np.exp(-2 * BETA * x))))
    return np.apply_along_axis(
        lambda h: 2 * BETA * (1 / (1 + np.exp(-2 * BETA * h))) * (1 - (1 / (1 + np.exp(-2 * BETA * h)))), 0, x)


def sigmoid_normalization(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return x / x
    max_value = max(x)
    min_value = min(x)
    return np.apply_along_axis(lambda h: (h - min_value) / (max_value - min_value), 0, x)

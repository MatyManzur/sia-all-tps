from typing import Callable
import numpy as np
from numpy._typing import NDArray

Activation_Function = Callable[[NDArray], NDArray]

BETA = 1


##LAS FUNCIONES DEBERIAN RECIBIR UN VECTOR COLUMNA CON LA SUMA DE LOS Wi*Xi

def identity(x: NDArray) -> NDArray:
    return x


def sign(x: NDArray) -> NDArray:
    return np.apply_along_axis(inclusive_sign, 0, x)


def inclusive_sign(x) -> int:
    return -1 + 2*(x >= 0)


def hiperbolic(x: NDArray) -> NDArray:
    return np.apply_along_axis(lambda h: BETA * np.tanh(h), 0, x)


def derivative_hiperbolic(x: NDArray) -> NDArray:
    return np.apply_along_axis(lambda h: BETA * (1 - (BETA * np.tanh(h)) ** 2), 0, x)


def sigmoid(x: NDArray) -> NDArray:
    return np.apply_along_axis(lambda h: 1 / (1 + np.exp(-2 * BETA * h)), 0, x)


def sigmoid_derivative(x: NDArray) -> NDArray:
    return np.apply_along_axis(
        lambda h: 2 * BETA * (1 / (1 + np.exp(-2 * BETA * h))) * (1 - (1 / (1 + np.exp(-2 * BETA * h)))), 0, x)

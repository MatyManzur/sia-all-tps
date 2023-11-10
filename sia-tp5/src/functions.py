from typing import Callable
import numpy as np
from numpy._typing import NDArray

Activation_Function = Callable[[NDArray|float], NDArray]
Normalization_Function = Callable[[NDArray|float], NDArray | float]

BETA = 1


##LAS FUNCIONES DEBERIAN RECIBIR UN VECTOR COLUMNA CON LA SUMA DE LOS Wi*Xi


def identity(x: NDArray | float) -> NDArray:
    return x


def derivative_identity(x: NDArray | float) -> NDArray:
    return np.ones_like(x)


def identity_normalization(x: NDArray | float) -> NDArray | float:
    return x


def sign(x: NDArray | float) -> NDArray | float:
    return -1 + 2 * (x >= 0)

def sign_derivative(x: NDArray | float) -> NDArray | float:
    return np.ones_like(x)

def sign_normalization(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return x / abs(x)
    avg = np.average(x)
    return np.sign(x - avg)

def inclusive_sign(x) -> int:
    return -1 + 2 * (x >= 0)


def hiperbolic(x: NDArray | float) -> NDArray | float:
    return BETA * np.tanh(x)


def hiperbolic_derivative(x: NDArray | float) -> NDArray | float:
    return BETA * (1 - (BETA * np.tanh(x)) ** 2)


def hiperbolic_normalization(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return x / abs(x)
    max_value = np.max(x)
    min_value = np.min(x)
    return 2 * (x - min_value) / (max_value - min_value) - 1


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-2 * BETA * x))


def sigmoid_derivative(x: NDArray | float) -> NDArray | float:
    return (2 * BETA * np.exp(-2 * BETA * x))/(1 + np.exp(-2 * BETA * x)) **2



def sigmoid_normalization(x: NDArray | float) -> NDArray | float:
    if isinstance(x, float) or isinstance(x, int):
        return x / abs(x)
    max_value = np.max(x)
    min_value = np.min(x)
    return (x - min_value) / (max_value - min_value)

import random

from typing import Tuple
import numpy as np
import numpy.random

from functions import *

from src.layer import Layer

LEARNING_CONSTANT = 0.1

DATA = [
    ((-1, -1), -1),
    ((-1, 1), -1),
    ((1, -1), -1),
    ((1, 1), 1)
]
# DATA = [
#     ((-1, -1), -1),
#     ((-1, 1), 1),
#     ((1, -1), 1),
#     ((1, 1), -1)
# ]
N = len(DATA[0][0])


def extract_in_out_from_data(i) -> Tuple[NDArray, float]:
    (inputs, expected_out) = DATA[i]
    inputs = np.array(inputs)
    inputs = np.insert(inputs, 0, [1]).T
    return inputs, expected_out


def compute_error(data, layer):
    sum = 0
    for i in range(len(data)):
        (X, ζ) = extract_in_out_from_data(i)
        O = layer.forward(X)
        sum += abs(ζ - O)
    return sum


def simple_step_perceptron():
    random.seed(123456789)
    numpy.random.seed(123456789)
    weights_at_min = None
    activation_fun = sign
    layer = Layer(N, 1, activation_fun)  # inicializa random
    min_error = float('inf')
    limit = 10000
    i = 0
    while min_error > 0 and i < limit:

        (inputs, expected_out) = extract_in_out_from_data(random.randint(0, len(DATA)-1))
        actual_out = layer.forward(inputs)

        delta_w = LEARNING_CONSTANT * (expected_out - actual_out) * inputs
        layer.weights += delta_w
        error = compute_error(DATA, layer)
        if error < min_error:
            min_error = error
            weights_at_min = layer
        i += 1
        print(i, end='\r')
    print()
    return weights_at_min


if __name__ == '__main__':
    w = simple_step_perceptron()
    print(f"{w.weights}")
    b = w.weights[0][0]
    x = w.weights[0][1]
    y = w.weights[0][2]
    print(f"Y = {x/-y} * X + {b/-y}")

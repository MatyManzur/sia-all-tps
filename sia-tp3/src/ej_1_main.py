import json
import random
from typing import Tuple

import numpy as np

from src.layer import Layer
from src.functions import *


DATA_AND = [
    ((-1, -1), -1),
    ((-1, 1), -1),
    ((1, -1), -1),
    ((1, 1), 1)
]
DATA_OR_EXC = [
     ((-1, -1), -1),
     ((-1, 1), 1),
     ((1, -1), 1),
     ((1, 1), -1)
 ]


def extract_in_out_from_data(data, i) -> Tuple[NDArray, float]:
    (inputs, expected_out) = data[i]
    inputs = np.array(inputs)
    inputs = np.insert(inputs, 0, [1]).T
    return inputs, expected_out


def step_compute_error(data, layer):
    sum = 0
    for i in range(len(data)):
        (X, ζ) = extract_in_out_from_data(data, i)
        O = layer.forward(X)
        sum += abs(ζ - O)
    return sum


# EJ1
def step_perceptron(limit=10000, data=None, learning_rate = 0.1):
    if data is None:
        data = DATA_AND
    result = {"weights": {}}
    weights_at_min = None
    activation_fun = sign
    layer = Layer(len(data[0][0]), 1, activation_fun)  # inicializa random
    min_error = float('inf')
    i = 0
    while min_error > 0 and i < limit:

        (inputs, expected_out) = extract_in_out_from_data(data, random.randint(0, len(data) - 1))
        actual_out = layer.forward(inputs)

        delta_w = learning_rate * (expected_out - actual_out) * inputs
        layer.weights += delta_w
        error = step_compute_error(data, layer)
        if error < min_error:
            min_error = error
            weights_at_min = layer
        if delta_w.all() != 0:
            result["weights"][f"iteration_{i}"] = {"w0": layer.weights[0][0], "w1": layer.weights[0][1], "w2": layer.weights[0][2]}
        i += 1
        print(i, end='\r')
    print()
    with open("results_step.json", "w") as outfile:
        json.dump(result, outfile)
    return weights_at_min

def print_data_from_line(w):
    b = w.weights[0][0]
    x = w.weights[0][1]
    y = w.weights[0][2]
    print(f"Y = {x / -y} * X + {b / -y}")

if __name__ == "__main__":
    W = step_perceptron(10000,DATA_AND)
    print_data_from_line(W)

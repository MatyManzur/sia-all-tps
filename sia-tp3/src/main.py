import json
import random
import csv
from sys import argv
from typing import Tuple
import numpy as np
import pandas as pd
import numpy.random

from functions import *
from layer import Layer

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


def step_compute_error(data, layer):
    sum = 0
    for i in range(len(data)):
        (X, ζ) = extract_in_out_from_data(i)
        O = layer.forward(X)
        sum += abs(ζ - O)
    return sum


def step_perceptron():
    result = {"weights": {}}
    weights_at_min = None
    activation_fun = sign
    layer = Layer(N, 1, activation_fun)  # inicializa random
    min_error = float('inf')
    limit = 10000
    i = 0
    while min_error > 0 and i < limit:

        (inputs, expected_out) = extract_in_out_from_data(random.randint(0, len(DATA) - 1))
        actual_out = layer.forward(inputs)

        delta_w = LEARNING_CONSTANT * (expected_out - actual_out) * inputs
        layer.weights += delta_w
        error = step_compute_error(DATA, layer)
        if error < min_error:
            min_error = error
            weights_at_min = layer
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
    print(f"Y = {x/-y} * X + {b/-y}")


EPSILON = 10 ** -1


def extract_in_out_ej2(data: NDArray, index=None):
    if index is None:
        index = random.randint(0, len(data) - 1)
    return np.insert(data[index][:3], 0, [1]).T, data[index][3]


def linear_compute_error(data, layer):
    sum = float(0)
    for i in range(len(data)):
        (inputs, expected) = extract_in_out_ej2(data, i)
        sum += (expected - layer.forward(inputs)) ** 2

    return sum / 2


def linear_perceptron(data: NDArray):
    weights_at_min = None
    activation_fun = sigmoid
    derivative_fun = sigmoid_derivative
    layer = Layer(len(data[0]) - 1, 1, activation_fun)  # inicializa random
    min_error = float('inf')
    limit = 1000
    i = 0
    while min_error > EPSILON and i < limit:

        (inputs, expected_out) = extract_in_out_ej2(data)
        actual_out = layer.forward(inputs)

        delta_w = LEARNING_CONSTANT * (expected_out - actual_out) * derivative_fun(
            layer.get_excitement(None)) * inputs
        layer.weights += delta_w
        error = linear_compute_error(data, layer)
        print(error, end='\r')
        if error < min_error:
            print()
            min_error = error
            weights_at_min = layer
        i += 1
        # print(i, end='\r')
    print()
    return weights_at_min,min_error


def print_data_from_line(w):
    b = w.weights[0][0]
    x = w.weights[0][1]
    y = w.weights[0][2]
    print(f"Y = {x / -y} * X + {b / -y}")


if __name__ == '__main__':
    random.seed()
    numpy.random.seed()

    dataframe = pd.read_csv(argv[1])
    dataarray = np.array(dataframe)
#    print(dataarray)
    w,error = linear_perceptron(dataarray)
    b = w.weights[0][0]
    x = w.weights[0][1]
    y = w.weights[0][2]
    z = w.weights[0][3]
    print(f"0=X*{x}+Y*{y}+Z*{z}+{b}")
    print(error)
    # print(f"Y = {x/-y} * X + {b/-y}")

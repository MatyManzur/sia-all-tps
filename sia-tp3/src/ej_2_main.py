import random
from sys import argv

import numpy.random
import pandas as pd

from src.functions import *
from src.layer import *

LEARNING_CONSTANT = 0.1

EPSILON = 10 ** -1


def extract_in_out_ej2(data: NDArray, index=None):
    if index is None:
        index = random.randint(0, len(data) - 1)
    return np.insert(data[index][:3], 0, [1]).T, data[index][3]


def linear_compute_error(data, layer):
    sum = float(0)
    for i in range(len(data)):
        (inputs, expected) = extract_in_out_ej2(data, i)
        sum += (layer.activation_function(expected) - layer.forward(inputs)) ** 2
        # sum += (expected - layer.forward(inputs)) ** 2

    return sum / 2

def linear_map_expected_output(data: NDArray):
    maxnum = max(abs(data[:,3]))
    # This should map the array to (0, 1)
    data[:, 3] = (data[:,3] / (2*maxnum) ) + 0.5
    return data


# EJ2
def linear_perceptron(data: NDArray):
    data = linear_map_expected_output(data)
    weights_at_min = None
    activation_fun = sigmoid
    derivative_fun = sigmoid_derivative
    layer = Layer(len(data[0]) - 1, 1, activation_fun, {
        "type": "momentum",
        "beta": 0
    })  # inicializa random
    min_error = float('inf')
    limit = 1000
    i = 0
    while min_error > EPSILON and i < limit:
        (inputs, expected_out) = extract_in_out_ej2(data)  # (X, ζ)
        actual_out = layer.forward(inputs)  # O = θ(h)
        delta_w = LEARNING_CONSTANT * (activation_fun(expected_out) - actual_out[0]) * derivative_fun(
            layer.excitement) * inputs  # ΔW = η * (θ(ζ)- O) * θ'(h) * X
        # delta_w = LEARNING_CONSTANT * (expected_out - actual_out[0]) * derivative_fun(
        #     layer.excitement) * inputs
        layer.weights += delta_w  # W = W + ΔW
        error = linear_compute_error(data, layer)  # E = (1/2)*Σ(θ(ζ)- O)²
        print(error, end='\r')
        if error < min_error:
            print()
            min_error = error
            weights_at_min = layer
        i += 1
        # print(i, end='\r')
    print()
    return weights_at_min, min_error



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
    w, error = linear_perceptron(dataarray)
    b = w.weights[0][0]
    x = w.weights[0][1]
    y = w.weights[0][2]
    z = w.weights[0][3]
    print(f"0=X*{x}+Y*{y}+Z*{z}+{b}")
    print(error)
    # print(f"Y = {x/-y} * X + {b/-y}")

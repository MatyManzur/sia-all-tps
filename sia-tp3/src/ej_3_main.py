import copy
import random
from typing import *

import numpy as np

from layer import *
from src.ej_2_main import EPSILON
from functions import *


# Un entrenamiento del neural net
def train_perceptron(neural_net: List[Layer], learning_constant: float, training_item: Tuple[NDArray, NDArray],
                     activation_func: Activation_Function, derivative_fun: Activation_Function):
    inputs, expected = training_item
    forward_propagation(neural_net, inputs)
    backpropagation(neural_net, derivative_fun, expected, inputs, learning_constant)

    return neural_net


def calculate_error_from_items(neural_net: List[Layer], items: List[Tuple[NDArray, NDArray]]) -> float:
    error_sum = 0
    for inputs, expected in items:
        outputs = forward_propagation(neural_net, inputs)
        error_sum += calculate_error(np.array(outputs), neural_net[0].activation_function(expected))
    return error_sum


DATA_OR_EXC = [
    ((-1, -1), (-1,)),
    ((-1, 1), (1,)),
    ((1, -1), (1,)),
    ((1, 1), (-1,))
]

ALGORITHM = 'online'
MINI_BATCH_SIZE = 2
LEARNING_CONSTANT = 0.1


def multilayer_perceptron(layers_neuron_count: List[int], act_func: Activation_Function,
                          deriv_func: Activation_Function,
                          data: List[Tuple[Tuple, Tuple]]):
    # La capa final tiene tantos nodos como outputs
    layers_neuron_count.append(len(data[0][1]))
    network = generate_layers(layers_neuron_count, len(data[0][0]), act_func)
    min_err = float('inf')
    w_min = None
    i = 0
    limit = 1

    while i < limit and min_err > EPSILON:
        if ALGORITHM == 'online':
            samples = random.sample(data, 1)
        elif ALGORITHM == 'mini-batch':
            samples = random.sample(data, MINI_BATCH_SIZE)
        elif ALGORITHM == 'batch':
            samples = data
        else:
            raise Exception('Invalid Algorithm!')

        for sample in samples:
            _sample = (np.array(sample[0]), np.array(sample[1]))
            train_perceptron(network, LEARNING_CONSTANT, _sample, act_func, deriv_func)
        consolidate_weights(network)
        reset_pending_weights(network)

        err = calculate_error_from_items(network, data)
        if err < min_err:
            min_err = err
            w_min = list(map(lambda layer: np.copy(layer.weights), network))

        i += 1
    print(min_err)

if __name__ == '__main__':
    multilayer_perceptron([2], identity,derivative_identity, DATA_OR_EXC)

import random
from typing import *

from functions import *
from layer import *


# Un entrenamiento del neural net
def train_perceptron(neural_net: List[Layer], learning_constant: float, training_item: Tuple[NDArray, NDArray],
                     activation_func: Activation_Function, derivative_fun: Activation_Function,
                     normalization_func: Activation_Function):
    inputs, expected = training_item
    forward_propagation(neural_net, inputs)
    backpropagation(neural_net, derivative_fun, normalization_func(expected), inputs, learning_constant)

    return neural_net


def calculate_error_from_items(neural_net: List[Layer], items: List[Tuple[Tuple, Tuple]], output_func) -> float:
    error_sum = 0
    for inputs, expected in items:
        sample, expected_output = np.array(inputs), np.array(expected)
        outputs = forward_propagation(neural_net, sample)
        error_sum += calculate_error(np.array(outputs), output_func(expected_output))
    return error_sum


DATA_OR_EXC = [
    ((-1, -1), (-1,)),
    ((-1, 1), (1,)),
    ((1, -1), (1,)),
    ((1, 1), (-1,))
]

ALGORITHM = 'online'
MINI_BATCH_SIZE = 2
LEARNING_CONSTANT = 0.3
EPSILON = 10 ** -4
BETA = 0.3


def multilayer_perceptron(layers_neuron_count: List[int], act_func: Activation_Function,
                          deriv_func: Activation_Function,
                          output_func,
                          data: List[Tuple[Tuple, Tuple]]):
    # La capa final tiene tantos nodos como outputs
    layers_neuron_count.append(len(data[0][1]))
    network = generate_layers(layers_neuron_count, len(data[0][0]), act_func, BETA)
    min_err = float('inf')
    w_min = None
    i = 0
    limit = 100000

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
            train_perceptron(network, LEARNING_CONSTANT, _sample, act_func, deriv_func, output_func)
        consolidate_weights(network)
        reset_pending_weights(network)

        err = calculate_error_from_items(network, data, output_func)
        if err < min_err:
            print(f"{i} - {err}")
            # sys.stdout.flush()
            min_err = err
            w_min = list(map(lambda layer: np.copy(layer.weights), network))
        else:
            print(f"{i} - {err} - {min_err}", end='\r')
        i += 1
    print()
    print(w_min)
    print(min_err)

    for i in range(len(network)):
        network[i].set_weights(w_min[i])

    print(forward_propagation(network, np.array([-1, -1])))
    print(forward_propagation(network, np.array([-1, 1])))
    print(forward_propagation(network, np.array([1, -1])))
    print(forward_propagation(network, np.array([1, 1])))


if __name__ == '__main__':
    multilayer_perceptron([6, 6], hiperbolic, hiperbolic_derivative, hiperbolic_normalization, DATA_OR_EXC)

# ()    ()   ()
# ()    ()   ()   ()
#       ()
#   3x3   2x4  1x3

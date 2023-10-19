import random
import time
from typing import *

from functions import *
from layer import *
from json import *

from data.ej3_digitos import DATA_DIGITOS
from data.ej3_digitos_par import DATA_DIGITOS_PAR
from data.ej3_digitos_ruido import DATA_DIGITOS_RUIDO


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
        outputs = np.array(forward_propagation(neural_net, sample))
        expected_output = output_func(np.reshape(expected_output, outputs.shape))
        error_sum += calculate_error(outputs, expected_output)
    return error_sum

def change_learning_rate(last_errors, learning_rate):
    should_change_rate = True
    for j in range(1, LEARNING_RATE_CHANGE_ITER):
        if last_errors[j-1] - last_errors[j] < CONSTANT_RATE_EPS:
            should_change_rate = False
            break

    if should_change_rate:
        return lambda x: x #+ 0.009    # Podemos parametrizar esto
    
    should_change_rate = True
    for j in range(1, LEARNING_RATE_CHANGE_ITER):
        if last_errors[j-1] - last_errors[j] > CONSTANT_RATE_EPS:
            should_change_rate = False
            break

    if should_change_rate:
        return lambda x: x #- learning_rate*0.0001    # Podemos parametrizar esto
    
    return lambda x: x

DATA_OR_EXC = [
    ((-1, -1), (-1,)),
    ((-1, 1), (1,)),
    ((1, -1), (1,)),
    ((1, 1), (-1,))
]

ALGORITHM = 'mini-batch'
MINI_BATCH_SIZE = 5
LEARNING_CONSTANT = 10 ** -2
EPSILON = 10 ** -2
BETA = 0
LIMIT = 100000
LAYERS = [64]
# Para cambiar el learning rate
LEARNING_RATE_CHANGE_ITER = 10
CONSTANT_RATE_EPS = 0.0001  


def multilayer_perceptron(layers_neuron_count: List[int], act_func: Activation_Function,
                          deriv_func: Activation_Function,
                          output_func,
                          data: List[Tuple[Tuple, Tuple]]):
    # La capa final tiene tantos nodos como outputs
    layers_neuron_count.append(len(data[0][1]))
    optimization = {}
    optimization['type'] ='momentum'
    optimization['beta'] = BETA

    network = generate_layers(layers_neuron_count, len(data[0][0]), act_func, optimization)
    min_err = float('inf')
    w_min = None
    i = 0

    start_time = time.time()

    learning_rate = LEARNING_CONSTANT

    if ALGORITHM == 'online':
        sample_size = 1
    elif ALGORITHM == 'mini-batch':
        sample_size = MINI_BATCH_SIZE
    elif ALGORITHM == 'batch':
        sample_size = len(data)
    else:
        raise Exception('Invalid Algorithm!')

    last_errors = []
    dump = []

    while i < LIMIT and min_err > EPSILON:
        samples = random.sample(data, sample_size)

        for sample in samples:
            _sample = (np.array(sample[0]), np.array(sample[1]))
            train_perceptron(network, learning_rate, _sample, act_func, deriv_func, output_func)
        consolidate_weights(network)
        reset_pending_weights(network)

        err = calculate_error_from_items(network, data, output_func)
        if i < LEARNING_RATE_CHANGE_ITER:
            last_errors.append(err)
        else:
            last_errors[i%LEARNING_RATE_CHANGE_ITER] = err

        if i%LEARNING_RATE_CHANGE_ITER == 0 and i != 0:
            learning_rate_change_func = change_learning_rate(last_errors, learning_rate)
            learning_rate = learning_rate_change_func(learning_rate)
            # print(learning_rate)
        
        # print(f"{i} - {err}", end='\r')

        curr = {
            'epoch': i,
            'error': err,
            'test_error': calculate_error_from_items(network, DATA_DIGITOS_RUIDO, output_func)
        }
        dump.append(curr)
        if err < min_err:
            # sys.stdout.flush()
            min_err = err
            w_min = list(map(lambda layer: np.copy(layer.weights), network))
        # else:
        #     print(f"{i} - {err} - {min_err}", end='\r')
        i += 1
    print()
    print(w_min)
    print(min_err)
    

    for i in range(len(network)):
        network[i].set_weights(w_min[i])

    # print(forward_propagation(network, np.array([-1, -1])))
    # print(forward_propagation(network, np.array([-1, 1])))
    # print(forward_propagation(network, np.array([1, -1])))
    # print(forward_propagation(network, np.array([1, 1])))

    for i, number in enumerate(DATA_DIGITOS_RUIDO):
        print(f"{i} - {forward_propagation(network, np.array(number[0]))}")

    end_time = time.time()
    print(f"Time: {end_time - start_time}")
    return {
        'iterations': dump
    }

    


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    data = multilayer_perceptron(LAYERS, sigmoid, sigmoid_derivative, sigmoid_normalization, DATA_DIGITOS)
    with open(f"sound-data.json", "w") as outfile:
        dump(data, outfile, indent=4)

# ()    ()   ()
# ()    ()   ()   ()
#       ()
#   3x3   2x4  1x3

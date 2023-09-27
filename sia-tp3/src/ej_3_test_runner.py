import json
import os
import random
import time
from sys import argv
from typing import Tuple

from data.ej3_digitos import DATA_DIGITOS
from data.ej3_digitos_par import DATA_DIGITOS_PAR
from data.xor_data import XOR_DATA
from ej_3_main import train_perceptron, calculate_error_from_items
from functions import *
from layer import *


def complete_confusion_matrix(test_data, network, output_func, confusion_matrix):
    DETERMINADO_EPSILON = 0.2
    for inputs, expected in test_data:
        values_to_run, expected_output = np.array(inputs), np.array(expected)
        outputs = np.array(forward_propagation(network, values_to_run))
        expected_output = output_func(np.reshape(expected_output, outputs.shape))
        expected_output_index = expected_output.argmax() # Este es el valor de la fila de la matriz
        real_output_index = outputs.argmax() # Ojo, en realidad con tomar el máximo, según Eugenia no alcanza
                                             # Porque podrías tener un 40% como máximo
        confusion_matrix[expected_output_index][real_output_index]+=1


def multilayer_perceptron(layers_neuron_count: List[int], act_func: Activation_Function,
                          deriv_func: Activation_Function, output_func, learning_data: List[Tuple[Tuple, Tuple]],
                          test_data: List[Tuple[Tuple, Tuple]], epsilon, limit, learning_constant, algorithm,
                          mini_batch_size, optimization_config=None):
    if optimization_config is None:
        optimization_config = {"type": "momentum", "beta": 0}
    # La capa final tiene tantos nodos como outputs
    layers_neuron_count.append(len(learning_data[0][1]))
    network = generate_layers(layers_neuron_count, len(learning_data[0][0]), act_func, optimization_config)
    min_err = float('inf')
    w_min = None
    min_err_generalization = float('inf')
    epoch_reached = 0
    i = 0

    output_data = {'iterations': []}

    while i < limit and min_err > epsilon:
        if algorithm == 'online':
            samples = random.sample(learning_data, 1)
        elif algorithm == 'mini-batch':
            samples = random.sample(learning_data, mini_batch_size)
        elif algorithm == 'batch':
            samples = learning_data
        else:
            raise Exception('Invalid Algorithm!')

        for sample in samples:
            _sample = (np.array(sample[0]), np.array(sample[1]))
            train_perceptron(network, learning_constant, _sample, act_func, deriv_func, output_func)
        consolidate_weights(network)
        reset_pending_weights(network)

        err = calculate_error_from_items(network, learning_data, output_func)
        test_error = calculate_error_from_items(network, test_data, output_func)
        output_data['iterations'].append({
            'epoch': i,
            'error': err,
            'test_error': test_error,
        })
        if err < min_err:
            epoch_reached = i
            min_err = err
            min_err_generalization = test_error
            w_min = list(map(lambda layer: layer.weights.tolist(), network))
        i += 1

    output_data['min_error'] = min_err
    output_data['min_error_generalization'] = min_err_generalization
    output_data['epoch_reached'] = epoch_reached
    output_data['weights'] = w_min
    return output_data


def run_test(config):
    layers = list(config['middle_layers_neurons'])

    if config['functions']['function_type'] == 'hiperbolic':
        activation_function = hiperbolic
        derivative_function = hiperbolic_derivative
        normalization_function = hiperbolic_normalization
    elif config['functions']['function_type'] == 'sigmoid':
        activation_function = sigmoid
        derivative_function = sigmoid_derivative
        normalization_function = sigmoid_normalization
    elif config['functions']['function_type'] == 'identity':
        activation_function = identity
        derivative_function = derivative_identity
        normalization_function = identity_normalization
    elif config['functions']['function_type'] == 'sign':
        activation_function = sign
        derivative_function = sign_derivative
        normalization_function = sign_normalization
    else:
        raise Exception('Invalid function type!')

    if config['data_set'] == 'DATA_DIGITOS':
        learning_data = [x for i, x in enumerate(DATA_DIGITOS) if i in config['learning_data_indexes']]
        test_data = [x for i, x in enumerate(DATA_DIGITOS) if i in config['test_data_indexes']]
    elif config['data_set'] == 'DATA_DIGITOS_PAR':
        learning_data = [x for i, x in enumerate(DATA_DIGITOS_PAR) if i in config['learning_data_indexes']]
        test_data = [x for i, x in enumerate(DATA_DIGITOS_PAR) if i in config['test_data_indexes']]
    elif config['data_set'] == 'XOR_DATA':
        learning_data = [x for i, x in enumerate(XOR_DATA) if i in config['learning_data_indexes']]
        test_data = [x for i, x in enumerate(XOR_DATA) if i in config['test_data_indexes']]
    else:
        raise Exception('Invalid data set!')

    epsilon = config['min_error']
    limit = config['max_iterations']
    learning_constant = config['learning_constant']
    algorithm = config['test']['training_type']
    mini_batch_size = config['test']['mini_batch_size']
    test_data = multilayer_perceptron(layers, activation_function, derivative_function, normalization_function,
                                      learning_data, test_data, epsilon, limit, learning_constant, algorithm,
                                      mini_batch_size, config['optimization'])

    test_data['config'] = {
        'layers': layers[:-1],  # remove last layer
        'data_set': config['data_set'],
        'functions': {'function_type': config['functions']['function_type']},
        'optimization': config['optimization'],
        'min_error': epsilon,
        'max_iterations': limit,
        'learning_constant': learning_constant,
        'test': {
            'training_type': algorithm,
            'mini_batch_size': mini_batch_size
        },
        'learning_data_indexes': config['learning_data_indexes'],
        'test_data_indexes': config['test_data_indexes'],
    }
    return test_data


if __name__ == '__main__':
    if len(argv) < 2:
        raise Exception('Invalid arguments!')
    config = json.load(open(argv[1], mode='r'))

    # get path from a file name
    # make directory
    path = os.path.dirname(os.path.abspath(config['file-name-prefix']))
    if not os.path.exists(path):
        os.makedirs(path)

    for test in config['tests']:
        print(f"=====Test: {test['name']}=====")
        start_test = time.time()
        results = []
        for i in range(config['iterations-per-test']):
            print(f"Iteration: {i}")
            # Run test
            test_results = run_test(test)
            results.append(test_results['min_error'])
            print(f"Reached minimum in epoch: {test_results['epoch_reached']}")
            # Save results of test
            with open(f"{config['file-name-prefix']}{test['name']}-{i}.json", "w") as outfile:
                json.dump(test_results, outfile, indent=4)
        end_test = time.time()
        print(f"\rElapsed time: {end_test - start_test}")
        print(f"Results: {results} - std: {np.std(results)} - mean: {np.mean(results)}")

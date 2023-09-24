import json
import time
import random
import sys
from sys import argv
from typing import Tuple
import numpy as np

from data.ej3_digitos import DATA_DIGITOS
from data.ej3_digitos_par import DATA_DIGITOS_PAR
from ej_3_main import train_perceptron, calculate_error_from_items
from functions import *
from layer import *


def multilayer_perceptron(layers_neuron_count: List[int], act_func: Activation_Function,
                          deriv_func: Activation_Function, output_func, learning_data: List[Tuple[Tuple, Tuple]],
                          test_data: List[Tuple[Tuple, Tuple]], beta, epsilon, limit, learning_constant, algorithm,
                          mini_batch_size):
    # La capa final tiene tantos nodos como outputs
    layers_neuron_count.append(len(learning_data[0][1]))
    network = generate_layers(layers_neuron_count, len(learning_data[0][0]), act_func, beta)
    min_err = float('inf')
    w_min = None
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
            'iteration': i,
            'error': err,
            'test_error': test_error,
        })
        print(f"\r{i} - {err} - {test_error}", end='')
        if err < min_err:
            min_err = err
            w_min = list(map(lambda layer: layer.weights.tolist(), network))
        i += 1

    print()
    output_data['min_error'] = min_err
    output_data['weights'] = w_min
    return output_data


def run_test(config):
    layers = config['middle_layers_neurons']

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
    else:
        raise Exception('Invalid function type!')

    if config['data_set'] == 'DATA_DIGITOS':
        learning_data = [x for i, x in enumerate(DATA_DIGITOS) if i in config['learning_data_indexes']]
        test_data = [x for i, x in enumerate(DATA_DIGITOS) if i in config['test_data_indexes']]
    elif config['data_set'] == 'DATA_DIGITOS_PAR':
        learning_data = [x for i, x in enumerate(DATA_DIGITOS_PAR) if i in config['test_data_indexes']]
        test_data = [x for i, x in enumerate(DATA_DIGITOS_PAR) if i in config['test_data_indexes']]
    else:
        raise Exception('Invalid data set!')

    beta = config['momentum_beta_value']
    epsilon = config['min_error']
    limit = config['max_iterations']
    learning_constant = config['learning_constant']
    algorithm = config['test']['training_type']
    mini_batch_size = config['test']['mini_batch_size']
    test_data = multilayer_perceptron(layers, activation_function, derivative_function, normalization_function,
                                 learning_data, test_data, beta, epsilon, limit, learning_constant, algorithm, mini_batch_size)

    test_data['config'] = {
        'layers': layers,
        'data_set': config['data_set'],
        'functions': { 'function_type': config['functions']['function_type'] },
        'momentum_beta_value': beta,
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

    for test in config['tests']:
        print(f"=====Test: {test['name']}=====")
        start_test = time.time()
        results = []
        for i in range(config['iterations-per-test']):
            print(f"Iteration: {i}")
            sys.stdout.flush()
            # Run test
            test_results = run_test(test)
            results.append(test_results['min_error'])
            # Save results of test
            with open(f"{config['file-name-prefix']}{i}.json", "w") as outfile:
                json.dump(test_results, outfile)
        end_test = time.time()
        print(f"Elapsed time: {end_test - start_test}")
        print(f"Results: {results} - std: {np.std(results)} - mean: {np.mean(results)}")

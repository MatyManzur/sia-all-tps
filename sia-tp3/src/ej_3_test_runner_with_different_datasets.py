import json
import os
import random
import time
from sys import argv
from typing import Tuple

from data.huge_digitos import TESTING_DATASET
from data.ej3_digitos import DATA_DIGITOS
from data.ej3_digitos_par import DATA_DIGITOS_PAR
from data.ej3_digitos_ruido import DATA_DIGITOS_RUIDO
from data.ej3_training_set_ruido import TRAINING_SET_RUIDO
from data.ej3_overtraining import DATA_DIGITOS_OVER
from data.xor_data import XOR_DATA
from ej_3_main import train_perceptron, calculate_error_from_items
from functions import *
from layer import *

dataset_mapper = {
    'DATA_DIGITOS': DATA_DIGITOS,
    'DATA_DIGITOS_PAR': DATA_DIGITOS_PAR,
    'DATA_DIGITOS_RUIDO': DATA_DIGITOS_RUIDO,
    'XOR_DATA': XOR_DATA,
    'TRAINING_SET_RUIDO': TRAINING_SET_RUIDO,
    'DATA_DIGITOS_OVER': DATA_DIGITOS_OVER,
    'TESTING_DATASET': TESTING_DATASET
}

def class_metrics_info(test_data, network):
    CLASS_DISTINCTION_EPS = 0.3 # SI ESTO ES MUY BAJO LA ACCURACY SIEMPRE TE DA COMO MÍNIMO 9
    different_classes = [input_output[1] for input_output in test_data]
    values_of_classes = []  # Se van a ir guardando en el orden en el que aparecen en la data
    actual_outputs = []
    for in_out_values in test_data:
        actual_outputs.append(np.array(forward_propagation(network, np.array(in_out_values[0]))))
    for particular_class in different_classes:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index, in_out_values in enumerate(test_data):
            # Si estamos hablando de la misma clase
            if np.array_equal(particular_class, in_out_values[1]):
                # tp o fn
                true_positive = True
                for i, outpt in enumerate(actual_outputs[index]):
                    if abs(outpt - particular_class[i]) > CLASS_DISTINCTION_EPS: # En alguno no me dio lo que pretendía
                        true_positive = False
                        fn+=1
                        break
                if true_positive:
                    tp+=1
            else:
                # fp o tn
                false_positive = True
                for i, outpt in enumerate(actual_outputs[index]):
                    if abs(outpt - particular_class[i]) > CLASS_DISTINCTION_EPS: # En alguno no me dio igual que la clase
                        false_positive = False
                        tn+=1
                        break
                if false_positive:
                    fp+=1
        values_of_classes.append({"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return values_of_classes


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
            train_perceptron(network, learning_constant, _sample, act_func, deriv_func)
        consolidate_weights(network)
        reset_pending_weights(network)

        err = calculate_error_from_items(network, learning_data)
        # test_error = calculate_error_from_items(network, test_data)
        
        # class_metrics_test = class_metrics_info(test_data, network)
        class_metrics_training = class_metrics_info(learning_data, network)

        output_data['iterations'].append({
            'epoch': i,
            'error': err / len(learning_data),
            # 'test_error': test_error / len(test_data),
            # 'class_metrics_test': class_metrics_test,
            'class_metrics_train': class_metrics_training,
        })
        if err < min_err:
            epoch_reached = i
            min_err = err
            # min_err_generalization = test_error
            w_min = list(map(lambda layer: layer.weights.tolist(), network))
        i += 1

    output_data['min_error'] = min_err
    output_data['min_error_generalization'] = min_err_generalization
    output_data['epoch_reached'] = epoch_reached
    output_data['weights'] = w_min
    result = list(map(lambda x: forward_propagation(network, np.array(x[0])), test_data))
    correct = 0.0
    for i, x in enumerate(result):
        # print(f"Expected: {test_data[i][1].index(max(test_data[i][1]))} - Got: {np.argmax(x)}")
        correct += 1 if test_data[i][1].index(max(test_data[i][1])) == np.argmax(x) else 0
    print(correct / len(test_data))
    return output_data


def apply_noise(noise_percentage, data):
    list = [*data]
    for i in range(len(data)):
        if random.random() < noise_percentage:
            list[i] = round(random.random(), 2)
    return list

def apply_bit_flip(noise_percentage, data):
    list = [*data]
    for i in range(len(data)):
        if random.random() < noise_percentage:
            list[i] = 1 - list[i]
    return list

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

    learning_data = dataset_mapper[config['data_set']]
    test_data = dataset_mapper[config['test_dataset']]
    # test_data = [(apply_noise(config['noise_probability'], data), exp) for (data, exp) in dataset_mapper[config['data_set']]]
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

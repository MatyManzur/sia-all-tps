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
from src.crossvalidator import CrossValidator
from src.metrics import accuracy


# Ahora que lo pienso, para determinado output, si está dandote fp -> está demasiado confiado (overscaling)
# Si te da fn -> entonces te está diciendo "no sé" (esto es preferible)  
# Habría que utilizar los negativos para esto
def is_saying_nonsense(actual_output):
    CLASS_DISTINCTION_EPS = 0.3 
    greatest_value = max(actual_output)
    if(greatest_value<1-CLASS_DISTINCTION_EPS):
        return True
    greatest_value_index = actual_output.index(greatest_value)
    for index, value in enumerate(actual_output):
        if index != greatest_value_index and value > CLASS_DISTINCTION_EPS:
            return False
    # Significa que piensa que es otro
    return True


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
            if np.array_equal(particular_class,in_out_values[1]):
                # tp o fn
                true_positive = True
                for i, outpt in enumerate(actual_outputs[index]):
                    if abs(outpt - particular_class[i] > CLASS_DISTINCTION_EPS): # En alguno no me dio lo que pretendía
                        true_positive = False
                        fn+=1
                        break
                if true_positive:
                    tp+=1
            else:
                # fp o tn
                false_positive = True
                for i, outpt in enumerate(actual_outputs[index]):
                    if abs(outpt - particular_class[i] > CLASS_DISTINCTION_EPS): # En alguno no me dio igual que la clase
                        false_positive = False
                        tn+=1
                        break
                if false_positive:
                    fp+=1
        values_of_classes.append({"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return values_of_classes


def complete_confusion_matrix(test_data, network, confusion_matrix):
    DETERMINADO_EPSILON = 0.2
    for inputs, expected in test_data:
        values_to_run, expected_output = np.array(inputs), np.array(expected)
        outputs = np.array(forward_propagation(network, values_to_run))
        expected_output = np.reshape(expected_output, outputs.shape)
        expected_output_index = expected_output.argmax() # Este es el valor de la fila de la matriz
        real_output_index = outputs.argmax() # Ojo, en realidad con tomar el máximo, según Eugenia no alcanza
                                             # Porque podrías tener un 40% como máximo
        confusion_matrix[expected_output_index][real_output_index]+=1


def multilayer_perceptron(layers_neuron_count: List[int], act_func: Activation_Function,
                          deriv_func: Activation_Function, learning_data: List[Tuple[Tuple, Tuple]],
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
        test_error = calculate_error_from_items(network, test_data)
        
        class_metrics_test = class_metrics_info(test_data, network)
        class_metrics_training = class_metrics_info(learning_data, network)

        output_data['iterations'].append({
            'epoch': i,
            'error': err,
            'test_error': test_error,
            'class_metrics_test': class_metrics_test,
            'class_metrics_train': class_metrics_training,
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

    for i, number in enumerate(DATA_DIGITOS_PAR):
        print(f"{i} - {forward_propagation(network, np.array(number[0]))}")

    return output_data


def run_test(config, training_test_sets = None):
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
        data_set = DATA_DIGITOS
    elif config['data_set'] == 'DATA_DIGITOS_PAR':
        data_set = DATA_DIGITOS_PAR
    elif config['data_set'] == 'XOR_DATA':
        data_set = XOR_DATA
    else:
        raise Exception('Invalid data set!')

    unnormalized_results = np.array(list(map(lambda x: x[1], data_set)))
    # print(unnormalized_results)
    normalized_results = normalization_function(unnormalized_results)
    # print(normalized_results)
    normalized_data = list(map(lambda x: (x[1][0], normalized_results[x[0]]), enumerate(data_set)))

    if training_test_sets is None:
        learning_data = [x for i, x in enumerate(normalized_data) if i in config['learning_data_indexes']]
        test_data = [x for i, x in enumerate(normalized_data) if i in config['test_data_indexes']]
    else:
        learning_data = training_test_sets[0]
        test_data = training_test_sets[1]

    epsilon = config['min_error']
    limit = config['max_iterations']
    learning_constant = config['learning_constant']
    algorithm = config['test']['training_type']
    mini_batch_size = config['test']['mini_batch_size']
    test_data = multilayer_perceptron(layers, activation_function, derivative_function,
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

def regular_test_runner():
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
        epochs = []
        for i in range(config['iterations-per-test']):
            print(f"Iteration: {i}")
            # Run test
            test_results = run_test(test)
            results.append(test_results['min_error'])
            print(f"Reached minimum in epoch: {test_results['epoch_reached']}")
            epochs.append(test_results['epoch_reached'])
            # Save results of test
            with open(f"{config['file-name-prefix']}{test['name']}-{i}.json", "w") as outfile:
                json.dump(test_results, outfile, indent=4)
        end_test = time.time()
        print(f"\rElapsed time: {end_test - start_test}")
        print(f"Results: {results} - std: {np.std(results)} - mean: {np.mean(results)}")
        print(f"Average epoch reached: {np.average(epochs)}")




if __name__ == '__main__':
    regular_test_runner()

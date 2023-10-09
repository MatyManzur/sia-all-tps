import random
from sys import argv
import json
import numpy.random
import pandas as pd

from src.functions import *
from src.layer import *
from src.crossvalidator import CrossValidator

LEARNING_CONSTANT = 0.01

EPSILON = 10 ** -10

FUNCTIONS_ARRAY = [
    [sigmoid, sigmoid_derivative, sigmoid_normalization],
    [hiperbolic, hiperbolic_derivative, hiperbolic_normalization],
    # [identity, derivative_identity, identity_normalization]
]


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


# EJ2
def linear_perceptron(training_data: NDArray, test_data: NDArray, function_array: List[Activation_Function],
                      learning_constant: float):
    weights_at_min = None
    activation_fun = function_array[0]
    derivative_fun = function_array[1]

    layer = Layer(len(training_data[0]) - 1, 1, activation_fun, {
        "type": "momentum",
        "beta": 0
    })

    min_error = float('inf')
    min_error_testing = float('inf')
    limit = 5000
    i = 0

    output_data = {'iterations': []}

    while min_error > EPSILON and i < limit:
        (inputs, expected_out) = extract_in_out_ej2(training_data)  # (X, ζ)
        actual_out = layer.forward(inputs)  # O = θ(h)
        delta_w = learning_constant * (expected_out - actual_out[0]) * derivative_fun(
            layer.excitement) * inputs  # ΔW = η * (θ(ζ)- O) * θ'(h) * X

        layer.add_pending_weight(delta_w)
        layer.consolidate_weights()

        error_test = linear_compute_error(test_data, layer)  # E = (1/2)*Σ(θ(ζ)- O)²
        error_training = linear_compute_error(training_data, layer)

        output_data['iterations'].append([i + 1, error_training[0][0], error_test[0][0]])  ## epoch, training,test

        if error_test < min_error:
            min_error = error_training
            min_error_testing = error_test
            weights_at_min = layer
        i += 1

    return weights_at_min, min_error, min_error_testing, output_data


def cross_validate(dataarray: NDArray, iterations, function_array: List[Activation_Function], learning_constant):
    cross_validator = None
    if iterations != 0:
        cross_validator = CrossValidator(dataarray, iterations)
    min_error = float('inf')
    min_error_weights = None
    min_data = None
    output = None
    for _ in range(iterations):
        data = cross_validator.next()
        if data is None:
            break
        weights, final_error, min_error_testing, output = linear_perceptron(np.array(data[0]), np.array(data[1]), function_array,
                                                         learning_constant)
        if final_error < min_error:
            min_error = final_error
            min_data = data
            min_error_weights = weights
    if iterations == 0:
        weights, final_error, min_error_testing, output = linear_perceptron(np.array(dataarray), np.array(dataarray), function_array,
                                                         learning_constant)
        min_error_weights = weights
        min_error = final_error
        min_data = dataarray
    return min_error_weights, min_error, min_data, output


def function_test(dataarray: NDArray, iterations):
    output = []
    for function_array in FUNCTIONS_ARRAY:
        unnormalized_results = dataarray[:, 3]
        normalized_results = function_array[2](unnormalized_results)
        normalized_results = np.reshape(normalized_results, (len(normalized_results), 1))
        normalized_data = np.append(dataarray[:, :3], normalized_results, axis=1)

        weights, min_error, min_data, output_data = cross_validate(normalized_data, iterations, function_array, 0.1)
        output.append({
            'function': function_array[0].__name__,
            'min_error': min_error[0][0],
            # 'weights': weights.weights.tolist(),
            # 'data': min_data,
            'output': output_data,
            'cross_iterations': iterations
        })
    json.dump(output, open('./results/ej_2_function_test.json', 'w'), indent=4)


def learning_test(dataarray: NDArray, iterations):
    output = []
    learning_text = "{constant:.2f}"
    unnormalized_results = dataarray[:, 3]
    normalized_results = hiperbolic_normalization(unnormalized_results)
    normalized_results = np.reshape(normalized_results, (len(normalized_results), 1))
    normalized_data = np.append(dataarray[:, :3], normalized_results, axis=1)

    for constant in np.arange(0.01, 0.1, 0.02):
        weights, min_error, min_data, output_data = cross_validate(normalized_data, iterations, FUNCTIONS_ARRAY[0],
                                                                   constant)
        output.append({
            'learning_constant': learning_text.format(constant=constant),
            'min_error': min_error[0][0],
            'output': output_data['iterations'],
        })
    json.dump(output, open('./results/ej_2_learning_test.json', 'w'), indent=4)


def data_parting_test(data_array: NDArray):
    outputs = []

    for function_list in FUNCTIONS_ARRAY:
        unnormalized_results = data_array[:, 3]
        normalized_results = function_list[2](unnormalized_results)
        normalized_results = np.reshape(normalized_results, (len(normalized_results), 1))
        normalized_data = np.append(data_array[:, :3], normalized_results, axis=1)
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 100]:
            testing_data = normalized_data[:int(len(normalized_data) * i)]
            if i == 100:
                training_data = testing_data
            else:
                training_data = normalized_data[int(len(normalized_data) * i):]
            weights, final_error, min_error_testing, output = linear_perceptron(training_data, testing_data, function_list, 0.01)
            data = list(map(lambda x: list(x), training_data)), list(map(lambda x: list(x), testing_data))

            outputs.append({
                'function': function_list[0].__name__,
                'min_error': final_error[0][0],
                'min_error_testing': min_error_testing[0][0],
                'percentage': i,
                'data': data
            })
    json.dump(outputs, open('./results/ej_2_data_parting_test.json', 'w'), indent=4)


if __name__ == '__main__':
    random.seed()
    numpy.random.seed()

    dataframe = pd.read_csv(argv[1])
    dataarray = np.array(dataframe)

    function_test(dataarray, 2)
    # learning_test(dataarray, 2)
    # data_parting_test(dataarray)

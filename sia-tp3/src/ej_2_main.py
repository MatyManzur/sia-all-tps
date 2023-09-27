import random
from sys import argv
import json
import numpy.random
import pandas as pd

from src.functions import *
from src.layer import *
from src.crossvalidator import CrossValidator

LEARNING_CONSTANT = 0.1

EPSILON = 10 ** -10

FUNCTIONS_ARRAY = [[sigmoid, sigmoid_derivative, sigmoid_normalization],
                   [hiperbolic, hiperbolic_derivative, hiperbolic_normalization],
                   #[identity, derivative_identity, identity_normalization]
                   ]


def extract_in_out_ej2(data: NDArray, index=None):
    if index is None:
        index = random.randint(0, len(data) - 1)
    return np.insert(data[index][:3], 0, [1]).T, data[index][3]


def linear_compute_error(data, layer, normalization: Activation_Function):
    sum = float(0)
    for i in range(len(data)):
        (inputs, expected) = extract_in_out_ej2(data, i)
        sum += (normalization(expected) - layer.forward(inputs)) ** 2

    return sum / 2


# EJ2
def linear_perceptron(training_data: NDArray, test_data: NDArray, function_array: List[Activation_Function]):
    weights_at_min = None
    activation_fun = function_array[0]
    derivative_fun = function_array[1]
    normalization = function_array[2]

    layer = Layer(len(training_data[0]) - 1, 1, activation_fun, {
        "type": "momentum",
        "beta": 0
    })

    min_error = float('inf')
    limit = 10000
    i = 0

    output_data = {'iterations': []}

    while min_error > EPSILON and i < limit:
        (inputs, expected_out) = extract_in_out_ej2(training_data)  # (X, ζ)
        actual_out = layer.forward(inputs)  # O = θ(h)
        delta_w = LEARNING_CONSTANT * (normalization(expected_out) - actual_out[0]) * derivative_fun(
            layer.excitement) * inputs  # ΔW = η * (θ(ζ)- O) * θ'(h) * X

        layer.add_pending_weight(delta_w)
        layer.consolidate_weights()

        error_test = linear_compute_error(test_data, layer, normalization)  # E = (1/2)*Σ(θ(ζ)- O)²
        error_training = linear_compute_error(training_data, layer, normalization)

        output_data['iterations'].append([i+1, error_training[0][0], error_test[0][0]]) ## epoch, training,test

        if error_test < min_error:
            min_error = error_test
            weights_at_min = layer
        i += 1

    return weights_at_min, min_error, output_data


def cross_validate(dataarray: NDArray, iterations, function_array: List[Activation_Function]):
    cross_validator = CrossValidator(dataarray, iterations)
    min_error = float('inf')
    min_error_weights = None
    min_data = None
    output = None
    for _ in range(iterations):
        data = cross_validator.next()
        if data is None:
            break
        weights, final_error, output = linear_perceptron(np.array(data[0]), np.array(data[1]), function_array)
        if final_error < min_error:
            min_error = final_error
            min_data = data
            min_error_weights = weights
    return min_error_weights, min_error, min_data, output


def function_test(dataarray: NDArray, iterations):
    output = []
    for function_array in FUNCTIONS_ARRAY:

        data_copy = np.array(dataarray)

        unnormalized_results = dataarray[:, 3]
        normalized_results = function_array[2](unnormalized_results)
        normalized_results = np.reshape(normalized_results, (len(normalized_results), 1))
        normalized_data = np.append(dataarray[:, :3], normalized_results, axis=1)

        weights, min_error, min_data, output_data = cross_validate(dataarray, iterations, function_array)
        output.append({
            'function': function_array[0].__name__,
            'min_error': min_error[0][0],
            #'weights': weights.weights.tolist(),
            #'data': min_data,
            'output': output_data,
            'cross_iterations': iterations
        })
    json.dump(output, open('./results/ej_2_function_test.json', 'w'), indent=4)


if __name__ == '__main__':
    random.seed()
    numpy.random.seed()

    dataframe = pd.read_csv(argv[1])
    dataarray = np.array(dataframe)

    function_test(dataarray, 2)

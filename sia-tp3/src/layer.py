import numpy as np
from functions import Activation_Function
from typing import List
from numpy._typing import NDArray


class Layer:  # N neuronas, con M inputs
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: Activation_Function):
        self.weights = np.random.rand(num_neurons, num_inputs + 1)
        self.activation_function = activation_function
        self.output = None  # Aca se guarda el resultado despues de salir de la funcion de activacion
        self.excitement = None  # Aca se guarda el valor de la suma ponderada del ultimo input
        self.pending_weight = None

    #
    def forward(self, inputs):  # inputs[0] must be 1 for bias
        self.output = self.activation_function(self.get_excitement(inputs))
        return self.output

    # NxM * Mx1 = Nx1
    def get_excitement(self, inputs=None):
        if inputs is not None:
            self.excitement = np.matmul(self.weights, inputs)
        return self.excitement
    
    def set_pending_weights(self, delta: NDArray):
        self.weights = delta

    def consolidate_weights(self):
        self.weights += self.pending_weight

    def reset_weights(self):
        self.weights = np.subtract(self.weights, self.pending_weight)

def generate_layers(layer_neurons: List[int], initial_inputs: int, act_func: Activation_Function) -> List[Layer]:
    prev_value = initial_inputs
    neural_network = []
    for neuron_count in layer_neurons:
        neural_network.append(Layer(prev_value, neuron_count, act_func))
        prev_value = neuron_count
    return neural_network


def forward_propagation(layer_neurons: List[Layer], training_data: NDArray) -> List[Layer]:
    input = training_data
    for layer in layer_neurons:
        input = layer.forward(input)
    return layer_neurons

def backpropagation(layer_neurons: List[Layer], derivative_func: Activation_Function, expected_output: NDArray, input: NDArray, learning_bias: float) -> List[Layer]:
    # Nx1 * Nx1 = Nx1
    initial_gradient = np.multiply(expected_output - layer_neurons[-1].output, derivative_func(layer_neurons[-1].excitement))
    # Resultado: Nx1*1xM = NxM
    initial_delta = learning_bias * np.matmul(initial_gradient, np.transpose(layer_neurons[-2].output))
    layer_neurons[-1].set_pending_weights(initial_delta)
    previous_gradient = initial_gradient
    previous_layer = layer_neurons[-1]
    # reverse iteration
    for i in range(len(layer_neurons) - 2, -1, -1):
        # MxN.Nx1 = Mx1 => Nx1*Nx1 = Nx1
        # δ^m = θ'(h) * ((W^m+1)' * δ^m+1)
        delta = np.multiply(derivative_func(layer_neurons[i].excitement),
                            np.matmul((np.transpose(previous_weight_change.weights))[1:], previous_delta))
        curr_input = layer_neurons[i - 1].output if i != 0 else input
        # Nx1.1xM = NxM
        delta = learning_bias * (gradient * np.transpose(curr_input))
        layer.set_pending_weights(delta)
    return layer_neurons


def calculate_error(calculated_output: NDArray, expected_out: NDArray) -> float:
    return (1 / 2) * (np.sum(np.square(np.subtract(expected_out, calculated_output))))



def consolidate_weights(neural_net: List[Layer]):
    for layer in neural_net:
        layer.consolidate_weights()


def reset_pending_weights(neural_net: List[Layer]):
    for layer in neural_net:
        layer.reset_pending_weights()

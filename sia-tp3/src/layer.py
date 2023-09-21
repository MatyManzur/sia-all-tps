import numpy as np
from functions import Activation_Function
from typing import List
from numpy._typing import NDArray


class Layer:  # N neuronas, con M inputs
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: Activation_Function):
        self.weights = np.random.rand(num_neurons, num_inputs + 1)
        self.activation_function = activation_function
        self.output = None      # Aca se guarda el resultado despues de salir de la funcion de activacion
        self.excitement = None  # Aca se guarda el valor de la suma ponderada del ultimo input

    #
    def forward(self, inputs):  # inputs[0] must be 1 for bias
        self.output = self.activation_function(self.get_excitement(inputs))
        return self.output

    # NxM * Mx1 = Nx1
    def get_excitement(self, inputs=None):
        if inputs is not None:
            self.excitement = np.dot(self.weights, inputs)
        return self.excitement
    
    def add_weights(self, delta: NDArray):
        self.weights = np.sum(self.weights, delta)

def generate_layers(layer_neurons: List[int], initial_inputs: int, act_func: Activation_Function) -> List[Layer]:
    prev_value = initial_inputs
    neural_network = []
    for neuron_count in layer_neurons:
        neural_network.append(Layer(prev_value, neuron_count, act_func))
        prev_value = neuron_count
    return generate_layers

def train_excitement(layer_neurons: List[Layer], training_data: NDArray) -> List[Layer]:
    input = training_data
    for layer in layer_neurons:
        input = layer.forward(input)
    return layer_neurons

def backpropagation(layer_neurons: List[Layer], derivative_func: Activation_Function, expected_output: NDArray, input: NDArray, learning_bias: float) -> List[Layer]:
    # Nx1 * Nx1 = Nx1
    initial_gradient = np.multiply(expected_output - layer_neurons[-1].output, derivative_func(layer_neurons[-1].excitement))
    # Resultado: NxM, Nx1.1xM
    initial_delta = learning_bias * np.dot(initial_gradient, np.transpose(layer_neurons[-2].output))
    layer_neurons[-1].add_weights(initial_delta)
    previous_gradient = initial_gradient
    previous_layer = layer_neurons[-1]
    # reverse iteration
    for layer, i in layer_neurons[0:-1:-1]:
        # MxN.Nx1 = Mx1 => Nx1
        gradient = derivative_func(layer.excitement) * np.dot(np.transpose(previous_layer.weights), previous_gradient)
        prev_input = layer[i - 1].output if i != 0 else input
        delta = learning_bias * np.dot(gradient, np.transpose(prev_input))
        layer.add_weights(delta)
    return layer_neurons
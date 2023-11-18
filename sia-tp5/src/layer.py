import numpy as np
from src.functions import Activation_Function
from typing import List, Tuple
from numpy._typing import NDArray
from src.optimization import Optimizer

LOWER_BOUND = -0.5
UPPER_BOUND = 0.5


# 0 < BETA < 1 Para momentum
class Layer:  # N neuronas, con M inputs
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: Activation_Function):
        self.weights = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND,
                                         size=(num_neurons, num_inputs + 1))  # +1 for bias
        self.activation_function = activation_function
        # Aca se guarda el resultado despues de salir de la funcion de activacion
        self.output = None
        # Aca se guarda el valor de la suma ponderada del ultimo input
        self.excitement = None
        # Aca se guardan los cambios de los pesos que se aplicarán al llamar a consolidate_weights()
        self.pending_weight = np.zeros_like(self.weights)
        # Se guarda el cambio de pesos anterior para el calculo con momentum
        self.last_weight_change = np.zeros_like(self.weights)

    #
    def forward(self, inputs):  # inputs[0] must be 1 for bias
        self.output = self.activation_function(self.get_excitement(inputs))
        return self.output

    def test_forward(self, inputs):
        return self.activation_function(self.get_excitement(inputs))

    # NxM * Mx1 = Nx1
    def get_excitement(self, inputs=None):
        if inputs is not None:
            self.excitement = np.matmul(self.weights, np.array([inputs]).T)
        return self.excitement

    def add_pending_weight(self, weight_change: NDArray, ):
        self.pending_weight = self.pending_weight + weight_change

    def consolidate_weights(self):
        self.weights += self.pending_weight
        self.reset_pending_weights()

    def reset_pending_weights(self):
        self.pending_weight = np.zeros_like(self.weights)

    def set_weights(self, weights):
        self.weights = weights


def generate_layers(layer_neurons: List[int], initial_inputs: int, act_func: Activation_Function) -> List[Layer]:
    prev_value = initial_inputs
    neural_network = []
    for neuron_count in layer_neurons:
        neural_network.append(Layer(prev_value, neuron_count, act_func))
        prev_value = neuron_count
    return neural_network


def forward_propagation(layer_neurons: List[Layer], training_data: NDArray) -> NDArray:
    input_values = training_data
    for layer in layer_neurons:
        input_values = layer.forward(np.append([1], input_values))  # append bias
    return input_values


def backpropagation(layer_neurons: List[Layer], derivative_func: Activation_Function,
                    expected_output: NDArray, input: NDArray, epoch: int, optimizer: Optimizer) -> Tuple[List[Layer], NDArray]:
    error = np.array([expected_output]).T - layer_neurons[-1].output
    return backpropagation_from_error(layer_neurons, derivative_func, error, input, epoch, optimizer)


def backpropagation_from_error(layer_neurons: List[Layer], derivative_func: Activation_Function,
                               error: NDArray, input: NDArray, epoch: int, optimizer: Optimizer) -> Tuple[List[Layer], NDArray]:
    # δ^f = θ'(h) * (ζ- V^f) (Nx1 * Nx1 = Nx1)
    delta = np.multiply(error,
                        derivative_func(layer_neurons[-1].excitement))
    # ΔW^m = δ^m * (V^m-1) (Nx1*1xM = NxM) -> el -η * ΔW^m se aplica en el optimizador
    weight_change = np.matmul(delta, np.array([np.append(1, layer_neurons[-2].output)]))

    # guarda el ΔW para aplicarlo más adelante
    layer_neurons[-1].add_pending_weight(optimizer.get_weight_change(weight_change, len(layer_neurons) - 1, epoch))
    previous_delta = delta
    previous_layer = layer_neurons[-1]
    # reverse iteration
    for i in range(len(layer_neurons) - 2, -1, -1):
        # MxN.Nx1 = Mx1 => Nx1*Nx1 = Nx1
        # δ^m = θ'(h) * ((W^m+1)' * δ^m+1)
        delta = np.multiply(derivative_func(layer_neurons[i].excitement),
                            np.matmul((np.transpose(previous_layer.weights))[1:], previous_delta))
        curr_input = layer_neurons[i - 1].output if i != 0 else input
        # Nx1.1xM = NxM
        # ΔW^m = δ^m * (V^m-1)' -> el η * ΔW^m se aplica en el optimizador
        curr_input = np.append([1], curr_input)
        weight_change = (delta * np.tile(curr_input, (len(delta), 1)))
        layer_neurons[i].add_pending_weight(optimizer.get_weight_change(weight_change, i, epoch))
        previous_delta = delta
        previous_layer = layer_neurons[i]
    return layer_neurons, previous_delta


def calculate_error(calculated_output: NDArray, expected_out: NDArray) -> float:
    return (1 / 2) * (np.sum((expected_out - calculated_output) ** 2))


def consolidate_weights(neural_net: List[Layer]):
    for layer in neural_net:
        layer.consolidate_weights()


def reset_pending_weights(neural_net: List[Layer]):
    for layer in neural_net:
        layer.reset_pending_weights()

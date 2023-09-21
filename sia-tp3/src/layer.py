import numpy as np
from functions import Activation_Function


class Layer:  # N neuronas, con M inputs

    def __init__(self, num_inputs: int, num_neurons: int, activation_function: Activation_Function):
        self.weights = np.random.rand(num_neurons, num_inputs + 1)
        self.activation_function = activation_function
        self.excitement = None  # Aca se guarda el valor de la suma ponderada del ultimo input

    def forward(self, inputs):  # inputs[0] must be 1 for bias
        self.excitement = np.dot(self.weights, inputs)
        return self.activation_function(self.excitement)

    def get_excitement(self, inputs=None):
        if inputs is not None:
            self.excitement = np.dot(self.weights, inputs)
        return self.excitement

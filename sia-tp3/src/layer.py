import numpy as np
from functions import Activation_Function

class Layer: #N neuronas, con M inputs

    def __init__(self, num_inputs: int, num_neurons: int, activation_function: Activation_Function):
        self.weights = np.random.rand(num_neurons, num_inputs + 1)
        self.activation_function = activation_function

    def forward(self, inputs): # inputs[0] must be 1 for bias
        return self.activation_function(np.dot(self.weights, inputs))


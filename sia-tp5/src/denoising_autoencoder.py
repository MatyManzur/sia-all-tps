from src.autoencoder import Autoencoder
import json
from typing import List, Tuple, Dict
from src.layer import *
from src.functions import *
import random
import time
from src.optimization import Optimizer
from datetime import timedelta

NoiseFunc = Callable[[list[int]], list[int]]

class DenoisingAutoencoder(Autoencoder):
  def __init__(self, encoder_layers: List[int], latent_space_dim: int, decoder_layers: List[int],
                 data: List[Tuple[Tuple, Tuple]], activation_function: Activation_Function,
                 derivation_function: Activation_Function,
                 normalization_function: Normalization_Function, optimization: Optimizer,
                 noise_func: NoiseFunc):
    self.noise_func = noise_func
    super().__init__(encoder_layers, latent_space_dim, decoder_layers, data, activation_function, derivation_function, normalization_function, optimization)

  def train_perceptron(self, neural_net: List[Layer], training_item: Tuple[NDArray, NDArray]):
        inputs, expected = training_item
        noisy_inputs = self.noise_func(inputs)
        forward_propagation(neural_net, np.array([noisy_inputs]))
        backpropagation(neural_net, self.deriv_func, np.array([expected]), np.array([inputs]), self.i, self.optimization)
        return neural_net
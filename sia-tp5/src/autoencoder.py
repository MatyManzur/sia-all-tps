from typing import List, Tuple, Dict
from src.layer import *
from src.functions import *
import random
import time
from src.optimization import Optimizer
# TODO: pasar a un config
ALGORITHM = 'batch'
MINI_BATCH_SIZE = 5
LIMIT = 100000
LAYERS = [64]
# Para cambiar el learning rate
LEARNING_RATE_CHANGE_ITER = 10
CONSTANT_RATE_EPS = 0.0001


class Autoencoder:
    def __init__(self, encoder_layers: List[int], latent_space_dim: int, decoder_layers: List[int],
                 data: List[Tuple[Tuple, Tuple]], activation_function: Activation_Function,
                 derivation_function: Activation_Function,
                 normalization_function: Normalization_Function, optimization: Optimizer):
        self.latent_layer_index = len(encoder_layers)
        layers_dimensions = encoder_layers + [latent_space_dim] + decoder_layers + [len(data[0][1])]
        self.network = generate_layers(layers_dimensions, len(data[0][0]), activation_function)
        self.min_err = float('inf')
        self.w_min = None
        self.i = 0
        self.last_errors = []
        unnormalized_results = np.array(list(map(lambda x: x[1], data)))
        normalized_results = normalization_function(unnormalized_results)
        self.normalized_data = list(map(lambda x: (x[1][0], normalized_results[x[0]]), enumerate(data)))
        self.act_func = activation_function
        self.deriv_func = derivation_function

        self.optimization = optimization

        if ALGORITHM == 'online':
            self.sample_size = 1
        elif ALGORITHM == 'mini-batch':
            self.sample_size = MINI_BATCH_SIZE
        elif ALGORITHM == 'batch':
            self.sample_size = len(self.normalized_data)
        else:
            raise Exception('Invalid Algorithm!')

    def __train_perceptron(self, neural_net: List[Layer], training_item: Tuple[NDArray, NDArray]):
        inputs, expected = training_item
        forward_propagation(neural_net, inputs)
        backpropagation(neural_net, self.deriv_func, expected, inputs, self.i, self.optimization)
        return neural_net

    def __train_step(self):
        # Agarramos un conjunto de samples según el algoritmo usado
        samples = random.sample(self.normalized_data, self.sample_size)
        # Entrena el autoencoder con cada sample, y luego actualiza los pesos
        for sample in samples:
            _sample = (np.array(sample[0]), np.array(sample[1]))
            self.__train_perceptron(self.network, _sample)
        consolidate_weights(self.network)
        reset_pending_weights(self.network)

        # Calcula el error
        err = self.__calculate_error_from_items()

        # Eta adaptativo
        if self.i < LEARNING_RATE_CHANGE_ITER:
            self.last_errors.append(err)
        else:
            self.last_errors[self.i % LEARNING_RATE_CHANGE_ITER] = err
        
        if self.i % LEARNING_RATE_CHANGE_ITER == 0 and self.i != 0:
            learning_rate_change_func = self.__change_learning_rate(self.last_errors, 0, 0) 
            self.optimization.set_learning_rate(learning_rate_change_func(self.optimization.learning_rate))

        if err < self.min_err:
            self.min_err = err
            self.w_min = list(map(lambda layer: np.copy(layer.weights), self.network))
        self.i += 1

    def train(self, step_count: int, min_err_threshold: float):
        start_time = time.time()
        while self.i < step_count and self.min_err > min_err_threshold:
            self.__train_step()
        for i in range(len(self.network)):
            self.network[i].set_weights(self.w_min[i])
        end_time = time.time()
        print(f"Number of steps: {self.i}")
        print(f"Min error: {self.min_err}")
        print(f"Time: {end_time - start_time}")

    def run_input(self, _input: Tuple) -> Tuple[NDArray, NDArray]:
        output = forward_propagation(self.network, np.array(_input))
        latent_output = self.network[self.latent_layer_index].output
        return output, latent_output

    def __calculate_error_from_items(self) -> float:
        error_sum = 0
        for inputs, expected in self.normalized_data:
            sample, expected_output = np.array(inputs), np.array(expected)
            outputs = np.array(forward_propagation(self.network, sample))
            expected_output = np.reshape(expected_output, outputs.shape)
            error_sum += calculate_error(outputs, expected_output)
        return error_sum / len(self.normalized_data)

    def __change_learning_rate(self, last_errors, a, b):
        should_change_rate = True
        for j in range(1, LEARNING_RATE_CHANGE_ITER):
            if last_errors[j - 1] - last_errors[j] < CONSTANT_RATE_EPS:
                should_change_rate = False
                break

        if should_change_rate:
            return lambda x: x + a    # Podemos parametrizar esto

        should_change_rate = True
        for j in range(1, LEARNING_RATE_CHANGE_ITER):
            if last_errors[j - 1] - last_errors[j] > CONSTANT_RATE_EPS:
                should_change_rate = False
                break

        if should_change_rate:
            return lambda x: x - x*b    # Podemos parametrizar esto

        return lambda x: x


class VAE:
    def __init__(self, data_size, encoderLayers: List[int], latent_space_dim: int, decoderLayers: List[int]):
        encoderLayers.append(latent_space_dim * 2)

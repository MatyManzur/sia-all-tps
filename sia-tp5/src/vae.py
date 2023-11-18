import json
from typing import List, Tuple, Dict

import numpy as np

from src.layer import *
from src.functions import *
import random
import time
from src.optimization import Optimizer
from datetime import timedelta

# TODO: pasar a un config
ALGORITHM = 'batch'
MINI_BATCH_SIZE = 5
LIMIT = 100000
LAYERS = [64]
# Para cambiar el learning rate
LEARNING_RATE_CHANGE_ITER = 10
CONSTANT_RATE_EPS = 0.0001


class VariationalAutoencoder: # TODO: xd falta el backpropagation
    def __init__(self, encoder_layers: List[int], latent_space_dim: int, decoder_layers: List[int],
                 data: List[Tuple[Tuple, Tuple]], activation_function: Activation_Function,
                 derivation_function: Activation_Function,
                 normalization_function: Normalization_Function, optimizer_encoder: Optimizer,
                 optimizer_decoder: Optimizer):
        self._latent_space_dim = latent_space_dim
        self.encoder = generate_layers(encoder_layers + [2 * latent_space_dim], len(data[0][0]), activation_function)
        self.decoder = generate_layers(decoder_layers + [len(data[0][1])], latent_space_dim, activation_function)

        self.min_err = float('inf')
        self.w_min = None
        self.i = 0
        self.last_errors = []
        unnormalized_results = np.array(list(map(lambda x: x[1], data)))
        normalized_results = normalization_function(unnormalized_results)
        self.normalized_data = list(map(lambda x: (x[1][0], normalized_results[x[0]]), enumerate(data)))
        self.act_func = activation_function
        self.deriv_func = derivation_function

        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder

        self.steps = []
        self.errors = []

        if ALGORITHM == 'online':
            self.sample_size = 1
        elif ALGORITHM == 'mini-batch':
            self.sample_size = MINI_BATCH_SIZE
        elif ALGORITHM == 'batch':
            self.sample_size = len(self.normalized_data)
        else:
            raise Exception('Invalid Algorithm!')

    def __forward_propagation_vae(self, input: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        encoder_output = forward_propagation(self.encoder, input)
        # Reparametrization trick
        mu_vec, sigma_vec = np.array_split(encoder_output, 2)
        epsilon = np.random.standard_normal()
        # z = μ + ε * σ
        z = mu_vec + epsilon * sigma_vec
        decoder_output = forward_propagation(self.decoder, z)
        return decoder_output, z, epsilon, mu_vec, sigma_vec
    """
    def sampling(args: tuple):
    # we grab the variables from the tuple
    z_mean, z_log_var = args
    print(z_mean)
    print(z_log_var)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon  # h(z)
    """

    def __train_perceptron(self, training_item: Tuple[NDArray, NDArray]):
        inputs, expected = training_item
        output, z, epsilon, mu, sigma = self.__forward_propagation_vae(inputs)
        # Decoder backpropagation
        _, last_delta_decoder = backpropagation(self.decoder, self.deriv_func, expected, z, self.i,
                                                self.optimizer_decoder)

        # Encoder backpropagation from reconstruction
        dz_dmu = np.ones([self._latent_space_dim,len(last_delta_decoder)])
        dz_dsigma = epsilon * np.ones([self._latent_space_dim,len(last_delta_decoder)])
        mu_error = np.dot(last_delta_decoder, dz_dmu)
        sigma_error = np.dot(last_delta_decoder, dz_dsigma)

        encoder_error = np.concatenate((mu_error, sigma_error), axis=0)

        backpropagation_from_error(self.encoder, self.deriv_func, encoder_error, inputs, self.i, self.optimizer_encoder)

        # Encoder backpropagation from regularization
        dL_dmu = mu
        dL_dv = 0.5 * (np.exp(sigma) - 1)
        encoder_loss_error = np.concatenate((dL_dmu, dL_dv), axis=0)
        backpropagation_from_error(self.encoder, self.deriv_func, encoder_loss_error, inputs, self.i,
                                   self.optimizer_encoder)

    def __train_step(self):
        # Agarramos un conjunto de samples según el algoritmo usado
        samples = random.sample(self.normalized_data, self.sample_size)
        # Entrena el autoencoder con cada sample, y luego actualiza los pesos
        for sample in samples:
            _sample = (np.array(sample[0]), np.array(sample[1]))
            self.__train_perceptron(_sample)

        consolidate_weights(self.encoder)
        consolidate_weights(self.decoder)


        # Calcula el error
        err = self.__calculate_error_from_items()

        # Eta adaptativo
        if self.i < LEARNING_RATE_CHANGE_ITER:
            self.last_errors.append(err)
        else:
            self.last_errors[self.i % LEARNING_RATE_CHANGE_ITER] = err

        if self.i % LEARNING_RATE_CHANGE_ITER == 0 and self.i != 0:
            learning_rate_change_func = self.__change_learning_rate(self.last_errors, 0, 0)
            self.optimizer_encoder.set_learning_rate(learning_rate_change_func(self.optimizer_encoder.learning_rate))
            self.optimizer_decoder.set_learning_rate(learning_rate_change_func(self.optimizer_decoder.learning_rate))

        if err < self.min_err:
            self.min_err = err
            self.w_min_encoder = list(map(lambda layer: np.copy(layer.weights), self.encoder))
            self.w_min_decoder = list(map(lambda layer: np.copy(layer.weights), self.decoder))
        self.i += 1

    def train(self, step_count: int, min_err_threshold: float, _print: bool = False):
        start_time = time.time()
        while self.i < step_count and self.min_err > min_err_threshold:
            if _print and self.i > 0:
                estimated_time = (time.time() - start_time) * (step_count - self.i) / self.i
                print(f"\rStep: {self.i} - Error: {self.min_err} - ETA: {timedelta(seconds=estimated_time)}", end='')
            self.__train_step()
        for i in range(len(self.encoder)):
            self.encoder[i].set_weights(self.w_min_encoder[i])
        for i in range(len(self.decoder)):
            self.decoder[i].set_weights(self.w_min_decoder[i])
        print()
        end_time = time.time()
        print(f"Number of steps: {self.i}")
        print(f"Min error: {self.min_err}")
        print(f"Time: {timedelta(seconds=end_time - start_time)}")

    def run_input(self, _input: Tuple) -> Tuple[NDArray, NDArray, NDArray]:
        decoder_output, z, epsilon, mu_vec, sigma_vec = self.__forward_propagation_vae(np.array(_input))
        return decoder_output, mu_vec, sigma_vec

    def output_from_latent_space(self, latent_space_values: Tuple) -> NDArray:
        return forward_propagation(self.decoder, np.array(latent_space_values))

    def __calculate_error_from_items(self) -> float:
        error_sum = 0
        for inputs, expected in self.normalized_data:
            sample, expected_output = np.array(inputs), np.array(expected)
            decoder_output, z, epsilon, mu_vec, sigma_vec = self.__forward_propagation_vae(np.array(sample))
            expected_output = np.reshape(expected_output, decoder_output.shape)
            # error_sum += calculate_error(decoder_output, expected_output) / N
            rec = 0.5 * np.mean((expected_output - decoder_output) ** 2)
            kl = -0.5 * np.sum(1 + sigma_vec - mu_vec ** 2 - np.exp(sigma_vec))
            error_sum += rec + kl
        #  kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return error_sum

    def __change_learning_rate(self, last_errors, a, b):
        should_change_rate = True
        for j in range(1, LEARNING_RATE_CHANGE_ITER):
            if last_errors[j - 1] - last_errors[j] < CONSTANT_RATE_EPS:
                should_change_rate = False
                break

        if should_change_rate:
            return lambda x: x + a  # Podemos parametrizar esto

        should_change_rate = True
        for j in range(1, LEARNING_RATE_CHANGE_ITER):
            if last_errors[j - 1] - last_errors[j] > CONSTANT_RATE_EPS:
                should_change_rate = False
                break

        if should_change_rate:
            return lambda x: x - x * b  # Podemos parametrizar esto

        return lambda x: x

    def save_weights(self, path: str):
        save_json = {"encoder": {}, "decoder": {}}

        for i in range(len(self.encoder)):
            save_json["encoder"][f"{i}"] = self.encoder[i].weights.tolist()

        for i in range(len(self.decoder)):
            save_json["decoder"][f"{i}"] = self.decoder[i].weights.tolist()

        with open(path, 'w') as f:
            json.dump(save_json, f)

    def load_weights(self, path: str):
        with open(path, 'r') as f:
            weights = json.load(f)

        for i in range(len(self.encoder)):
            self.encoder[i].set_weights(np.array(weights["encoder"][f"{i}"]))
        for i in range(len(self.decoder)):
            self.decoder[i].set_weights(np.array(weights["decoder"][f"{i}"]))

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
ALGORITHM = 'mini-batch'
MINI_BATCH_SIZE = 100
LIMIT = 100000
LAYERS = [64]
# Para cambiar el learning rate
LEARNING_RATE_CHANGE_ITER = 10
CONSTANT_RATE_EPS = 0.0001


class VariationalAutoencoder:
    def __init__(self, encoder_layers: List[int], latent_space_dim: int, decoder_layers: List[int],
                 data: List[Tuple[Tuple, Tuple]], activation_function: Activation_Function,
                 derivation_function: Activation_Function,
                 normalization_function: Normalization_Function, optimizer_encoder: Optimizer,
                 optimizer_decoder: Optimizer):
        self._latent_space_dim = latent_space_dim
        self.encoder = generate_layers(encoder_layers + [2 * latent_space_dim], len(data[0][0]), activation_function)
        self.decoder = generate_layers(decoder_layers + [len(data[0][1])], latent_space_dim, activation_function)

        self.min_err = float('inf')
        self.w_min_encoder = None
        self.w_min_decoder = None

        self.i = 0
        self.last_errors = []
        unnormalized_results = np.array(list(map(lambda x: x[1], data)))
        self.normalization_function = normalization_function
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

    def __forward_propagation_vae(self, inputs: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        encoder_output = forward_propagation(self.encoder, inputs)
        # Reparametrization trick
        mu_vec, sigma_vec = np.array_split(encoder_output, 2, axis=1)
        epsilon = np.random.standard_normal() # TODO: es un escalar o un vector?
        # epsilon = np.reshape(np.random.standard_normal(self._latent_space_dim),[self._latent_space_dim,1] )
        # z = μ + ε * σ
        z = mu_vec + np.multiply(epsilon,sigma_vec)
        # normalized_z = self.normalization_function(z)
        # decoder_output = forward_propagation(self.decoder, normalized_z)
        decoder_output = forward_propagation(self.decoder, z)
        return decoder_output, z, epsilon, mu_vec, sigma_vec

    def __train_perceptron(self, training_items: Tuple[NDArray, NDArray]):
        inputs, expected = training_items
        output, z, epsilon, mu, sigma = self.__forward_propagation_vae(inputs)
        # Decoder backpropagation
        _, last_delta_decoder = backpropagation(self.decoder, self.deriv_func, expected, z, self.i,
                                                self.optimizer_decoder)
        last_delta_decoder = last_delta_decoder.T
        last_delta_size = len(last_delta_decoder[0])

        # Encoder backpropagation from reconstruction
        dz_dmu = np.ones([last_delta_size, self._latent_space_dim])
        dz_dsigma = epsilon * np.ones([last_delta_size, self._latent_space_dim])
        dE_dmu = np.dot(last_delta_decoder, dz_dmu)
        dE_dsigma = np.dot(last_delta_decoder, dz_dsigma)

        encoder_error = np.concatenate((dE_dmu, dE_dsigma), axis=1).T

        backpropagation_from_error(self.encoder, self.deriv_func, encoder_error, inputs, self.i, self.optimizer_encoder)

        # Encoder backpropagation from regularization
        dL_dmu = mu
        dL_dsigma = 0.5 * (np.exp(sigma) - 1)
        encoder_loss_error = np.concatenate((dL_dmu, dL_dsigma), axis=1).T
        backpropagation_from_error(self.encoder, self.deriv_func, encoder_loss_error, inputs, self.i,
                                        self.optimizer_encoder)

        rec = 0.5 * np.mean((expected - output) ** 2)
        kl = -0.5 * np.sum(1 + sigma - mu ** 2 - np.exp(sigma))
        loss = rec + kl

        return loss

    def __train_step(self):
        # Agarramos un conjunto de samples según el algoritmo usado

        samples_0 = [list(letter[0]) for letter in self.normalized_data]
        samples_1 = [list(letter[1]) for letter in self.normalized_data]
        samples = (np.array(samples_0), np.array(samples_1))

        err = self.__train_perceptron(samples)

        consolidate_weights(self.encoder)
        consolidate_weights(self.decoder)

        # Eta adaptativo
        if self.i < LEARNING_RATE_CHANGE_ITER:
            self.last_errors.append(err)
        else:
            self.last_errors[self.i % LEARNING_RATE_CHANGE_ITER] = err

        if self.i % LEARNING_RATE_CHANGE_ITER == 0 and self.i != 0:
            learning_rate_change_func = self.__change_learning_rate(self.last_errors, 0, 0)
            self.optimizer_encoder.set_learning_rate(learning_rate_change_func(self.optimizer_encoder.learning_rate))
            self.optimizer_decoder.set_learning_rate(learning_rate_change_func(self.optimizer_decoder.learning_rate))

        if True or (err < self.min_err):
            self.min_err = err
            self.w_min_encoder = list(map(lambda layer: np.copy(layer.weights), self.encoder))
            self.w_min_decoder = list(map(lambda layer: np.copy(layer.weights), self.decoder))
        self.i += 1

    def train(self, step_count: int, min_err_threshold: float, _print: bool = False):
        start_time = time.time()
        while self.i < step_count and self.min_err > min_err_threshold:
            if self.i % 10 == 0 and self.i != 1:
                self.steps.append(self.i)
                self.errors.append(self.min_err)
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

    def run_input(self, _input: Tuple) -> Tuple[NDArray, NDArray]:
        decoder_output, z, epsilon, mu_vec, sigma_vec = self.__forward_propagation_vae(np.array([_input]))
        return decoder_output, z.T

    def output_from_latent_space(self, latent_space_values: Tuple) -> NDArray:
        return forward_propagation(self.decoder, np.array([latent_space_values]))

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

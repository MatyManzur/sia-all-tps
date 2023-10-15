import math
import random
from typing import List, Callable, Tuple, Dict

import numpy as np
from numpy._typing import NDArray

SimilarityFunction = Callable[[NDArray[float], NDArray[float]], float]


def euclidean_distance(first: NDArray[float], second: NDArray[float]) -> float:
    return np.linalg.norm(first - second)


def exponential_distance(first: NDArray[float], second: NDArray[float]) -> float:
    return math.exp(-1 * math.pow(euclidean_distance(first, second), 2))


class Kohonen:
    def __init__(self, k: int, input_size: int, max_iterations: int,
                 get_learning_rate: Callable[[int], float], distance_function: SimilarityFunction,
                 initial_radius: float, radius_change: Callable[[float, int], float],
                 standardized_data: List[NDArray[float]], random_initial_weights: bool):
        self.k = k
        self.initial_radius = initial_radius
        self.current_iteration = 0
        self.get_learning_rate = get_learning_rate
        self.distance_function = distance_function
        self.radius_change = radius_change
        self.max_iterations = max_iterations
        self.standardized_data = standardized_data

        if random_initial_weights:
            self.weights = np.random.rand(k, k, input_size)
        else:
            self.weights = np.zeros(shape=(k, k, input_size))
            for y in range(self.k):
                for x in range(self.k):
                    self.weights[x][y] = random.sample(standardized_data, 1)[0]

    def train(self):
        while self.current_iteration < self.max_iterations:
            _input = random.sample(self.standardized_data, 1)[0]
            self.__next(_input)

    def __next(self, input: NDArray[float]):
        most_similar = None
        most_similar_difference = 0
        (most_similar, most_similar_difference) = self.get_most_similar_neuron(input)

        radius = self.radius_change(self.initial_radius, self.current_iteration)

        self.__update_weights_in_neighbourhood(most_similar[0], most_similar[1], input, radius)
        self.current_iteration += 1
        return self.weights

    def get_most_similar_neuron(self, _input: NDArray[float]) -> Tuple[Tuple[int, int], float]:
        standarized_input = _input
        most_similar_difference = 0
        most_similar = None
        for y, rows in enumerate(self.weights):
            for x, col in enumerate(rows):
                aux = self.distance_function(standarized_input, col)  # usar distance function
                if most_similar_difference > aux or most_similar is None:
                    most_similar_difference = aux
                    most_similar = (x, y)
        return most_similar, most_similar_difference

    def __update_weights_in_neighbourhood(self, x: int, y: int, _input: NDArray[float], radius: float):
        rows, cols, weights = self.weights.shape
        eta = self.get_learning_rate(self.current_iteration)

        for i in range(max(0, x - math.ceil(radius)), min(rows, x + math.ceil(radius) + 1)):
            for j in range(max(0, y - math.ceil(radius)), min(cols, y + math.ceil(radius) + 1)):
                distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                if distance <= radius and distance != 0:
                    self.weights[j][i] += eta * (_input - self.weights[j][i])

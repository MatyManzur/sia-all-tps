from typing import List

import numpy as np
from numpy._typing import NDArray


def generate_hopfield_matrix(training_set: List[NDArray]):
    elements = np.column_stack(training_set)
    base = (1.0 / len(training_set[0])) * (np.dot(elements, np.transpose(elements)))
    np.fill_diagonal(base, 0)
    return base


def most_similar_pattern(hopfield: NDArray, consult: NDArray, max_iter: int):
    iteration = 0
    current = np.sign(hopfield.dot(consult))
    iterations = [(iteration, current, get_energy(hopfield, current))]
    previous = current
    while iteration < max_iter:
        iteration += 1
        current = np.sign(hopfield.dot(current))
        iterations.append((iteration, current, get_energy(hopfield, current)))
        if np.array_equal(current, previous):
            break
        previous = current
    return current, iteration, np.array_equal(current, consult), iterations

def get_energy(hopfield: NDArray, consult: NDArray):
    return -0.5 * consult.dot(hopfield.dot(consult.T))
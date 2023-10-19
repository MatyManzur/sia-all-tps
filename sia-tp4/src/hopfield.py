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
    while not np.array_equal(current, consult) and iteration < max_iter:
        current = np.sign(hopfield.dot(current))
        iteration += 1
    return current, iteration, np.array_equal(current, consult)

from typing import List

import numpy as np
from numpy._typing import NDArray


def generate_hopfield_matrix(training_set: List[NDArray]):
    elements = np.column_stack(training_set)
    return (1.0 / len(training_set)) * elements * np.transpose(elements)


def most_similar_pattern(hopfield: NDArray, consult: NDArray, max_iter: int):
    iteration = 0
    current = np.sign(hopfield * consult)
    while current != consult and iteration < max_iter:
        current = np.sign(hopfield * consult)
        iteration += 1
    return current

import random
from typing import Tuple, List
import numpy as np

class CrossValidator:

    def __init__(self, data, iterations):
        self.data = data
        np.random.seed()
        np.random.shuffle(self.data)
        self.iterations = iterations
        self.sample_size = len(data) // iterations
        self.current_iteration = 0

    def next(self) -> Tuple[List, List] | None:
        if self.current_iteration == self.iterations:
            return None
        start = self.current_iteration * self.sample_size
        end = start + self.sample_size
        if(self.current_iteration == 0):
            return self.data[end:], self.data[start:end] # training, test
        self.current_iteration += 1
        return self.data[:start] + self.data[end:], self.data[start:end]  # training, test

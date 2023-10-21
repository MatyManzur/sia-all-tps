import numpy as np
from sklearn.decomposition import PCA
from src.standardization import z_score
import pandas as pd
import random
from functions import *
from typing import List

CSV_FILE = 'data/europe.csv'
LOWER_BOUND = 0
UPPER_BOUND = 1


class SimplePerceptron:

    def __init__(self, num_inputs: int, activation_function: Activation_Function):
        self.weights = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND,
                                         size=num_inputs)
        self.activation_function = activation_function

    def forward(self, inputs):
        return self.activation_function(np.dot(self.weights, inputs))


def oja(data_list: List[NDArray], column_count: int):
    perceptron = SimplePerceptron(column_count, identity)
    np.random.seed()
    random.seed()
    initial_learning_rate = 0.087
    max_epoch = 100
    for i in range(1,max_epoch):
        learning_rate = initial_learning_rate / i
        for sample in data_list:
            output = perceptron.forward(sample)
            # delta_w = learning_rate * output * sample #Sin aproximar la norma
            # w_aux = perceptron.weights + delta_w
            # perceptron.weights = w_aux / np.linalg.norm(w_aux)
            delta_w = learning_rate *((sample * output) - ((output ** 2) * perceptron.weights))  # Aproximando la norma
            perceptron.weights += delta_w

    return perceptron.weights


if __name__ == '__main__':
    data = pd.read_csv(CSV_FILE)
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()

    loadings = oja(data_array, len(columns))
    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(data_array)  # paises en la nueva base de componentes pcpales
    print(pca.components_)
    print(loadings)

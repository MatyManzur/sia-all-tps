import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import random
from src.standardization import z_score
from functions import *
from typing import List

CSV_FILE = 'data/europe.csv'
LOWER_BOUND = 0
UPPER_BOUND = 1


class SimplePerceptron:

    def __init__(self, num_inputs: int, activation_function: Activation_Function):
        self.weights = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND,
                                         size=(num_inputs + 1))  # +1 for bias
        self.activation_function = activation_function
        self.output = None  # Aca se guarda el resultado despues de salir de la funcion de activacion
        self.excitement = None  # Aca se guarda el valor de la suma ponderada del ultimo input

    def forward(self, inputs):  # inputs[0] must be 1 for bias
        self.output = self.activation_function(self.get_excitement(inputs))
        return self.output

    def get_excitement(self, inputs=None):
        if inputs is not None:
            self.excitement = np.matmul(self.weights, np.array([inputs]).T)
        return self.excitement


def oja(data_list: List[NDArray],column_count:int):

    perceptron = SimplePerceptron(column_count, sigmoid)
    np.random.seed()
    initial_learning_rate = learning_rate = 0.5
    max_iter = 10000
    i = 0
    error = float('inf')
    min_error = 10**-10
    while error > min_error and i < max_iter:
        sample = data_list[random.randint(0, len(data_list)-1)]
        sample = np.insert(sample, 0, [1]).T  # Ponemos el 1 al principio para el bias
        output = perceptron.forward(sample)
        # delta_w = learning_rate * output * sample Sin aproximar la norma
        # w_aux = perceptron.weights + delta_w
        # perceptron.weights = w_aux / np.linalg.norm(w_aux)
        delta_w = learning_rate * (output * sample - (output ** 2 * perceptron.weights))  # Aproximando la norma
        perceptron.weights += delta_w
        error = np.linalg.norm(delta_w)
        print(f'Error {error} : iter {i}----\n')
        i += 1
    return perceptron.weights


if __name__ == '__main__':
    data = pd.read_csv(CSV_FILE)
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()

    loadings = oja(data_array,len(columns))
    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(data_array)  # paises en la nueva base de componentes pcpales
    print(pca.components_)  # array de autovectores
    print(loadings)


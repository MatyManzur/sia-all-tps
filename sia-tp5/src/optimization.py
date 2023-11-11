import numpy as np
from numpy._typing import NDArray


class Optimizer:
    def __init__(self,learning_rate: float = 0.1):
        self.learning_rate = learning_rate

    def get_weight_change(self, gradient: NDArray[float], layer: int, epoch: int):
        raise NotImplementedError()

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate


class MomentumOptimizer(Optimizer):
    def __init__(self, amount_of_layers: int, learning_rate: float = 0.1, beta: float = 0.3):
        super().__init__(learning_rate)
        self.beta = beta
        self.last_weights_changes = [0]*amount_of_layers

    def get_weight_change(self, gradient: NDArray[float], layer: int, _) -> NDArray[float]:
        self.last_weights_changes[layer] =np.array(self.learning_rate * gradient + self.beta * self.last_weights_changes[layer])
        return self.last_weights_changes[layer]


class AdamOptimizer(Optimizer):

    def __init__(self,amount_of_layers:int, alpha:float=0.001, beta_1:float=0.9, beta_2:float=0.999, epsilon:float=1e-8):
        super().__init__(alpha)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = [0]*amount_of_layers
        self.rmsprop = [0]*amount_of_layers

    def get_weight_change(self, gradient: NDArray[float], layer: int, epoch: int):
        self.momentum[layer] = self.beta_1 * self.momentum[layer] + (1 - self.beta_1) * gradient
        self.rmsprop[layer] = self.beta_2 * self.rmsprop[layer] + (1 - self.beta_2) * gradient ** 2
        m_hat = self.momentum[layer] / (1 - self.beta_1 ** (epoch+1))
        r_hat = self.rmsprop[layer] / (1 - self.beta_2 ** (epoch+1))
        return self.learning_rate * (m_hat / (np.sqrt(r_hat) + self.epsilon))
import numpy as np
from Model.layers.linear import LinearLayer
from typing import List

class AdaGradSolver:
    """Implements the AdaGrad optimization algorithm.
    
    AdaGrad Algorithm:
    Adaptive Gradient Algorithm (AdaGrad) scales learning rates individually for each parameter.
    W = W - lr * mean(G) / sqrt(G_squared_accumulated + epsilon)

    where G_squared_accumulated = G_squared_accumulated + mean(square(gradient))

    Attributes:
        learning_rate (float): Learning rate of the network.
        modules (List[Type]): List of layers to be optimized. 
                               All layers except the input layer should inherit from `LinearLayer`.
        epsilon (float, optional): Small constant to prevent division by zero. Defaults to 1e-8.
    """
    def __init__(self, learning_rate: float, modules: List[LinearLayer], epsilon: float=1e-8):
        self.learning_rate = learning_rate
        self.modules = modules
        self.epsilon = epsilon
        for module in modules:
            module.G_squared_accumulated = 0

    def step(self):
        """Updates the weights of the modules using the AdaGrad algorithm."""
        for module in self.modules:
            module.G_squared_accumulated += module.G.mean(axis=0)**2
            module.W -= self.learning_rate * module.G.mean(axis=0) / (np.sqrt(module.G_squared_accumulated) + self.epsilon) 
            pass
        pass
    pass


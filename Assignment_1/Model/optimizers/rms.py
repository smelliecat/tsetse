import numpy as np
from typing import List
from layers.linear import LinearLayer

class RMSPropSolver:
    """
    Implements the RMSProp optimization algorithm.
    
    RMSProp Algorithm:
    RMSProp maintains a moving average of the squared gradient for each parameter.
    W = W - lr * mean(G) / sqrt(G_squared_moving_avg + epsilon)
    G_squared_moving_avg = beta * G_squared_moving_avg + (1 - beta) * mean(square(gradient))
    W -= learning_rate * gradient / sqrt(G_squared_moving_avg + epsilon)

    Parameters:
    - lr (float): Learning rate.
    - modules (List): List of layers in the model, excluding the input layer.
                       All layers should have a parent class of LinearLayer.
    - beta (float): Decay factor for the moving average.
    - epsilon (float): Small constant to prevent division by zero.
    """
    def __init__(self, learning_rate: float, modules:List[LinearLayer], beta: float=0.9, epsilon: float=1e-8):
        self.learning_rate = learning_rate
        self.modules = modules
        self.beta = beta
        self.epsilon = epsilon
        for module in self.modules:
            module.G_squared_moving_avg = 0

    def step(self):
        """
        Perform a single optimization step, updating the parameters of all layers in 'modules'.
        """
        for module in self.modules:
            module.G_squared_moving_avg = self.beta * module.G_squared_moving_avg + (1 - self.beta) * module.G.mean(axis=0)**2
            module.W -= self.lr * module.G.mean(axis=0) / (np.sqrt(module.G_squared_moving_avg) + self.epsilon)
            pass
        pass
    pass
        
        
import numpy as np
from typing import List
from layers.linear import LinearLayer
class SGDSolver:
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
    
    Algorithm Description:
    SGD updates each parameter (W) based on the gradient (G) of the objective function 
    with respect to that parameter. The formula for the parameter update is:
    
    W = W - lr * mean(G)
    
    where lr is the learning rate and mean(G) is the mean of the gradients.
    
    Parameters:
    - learning_rate (float): Learning rate.
    - modules (List): List of layers in the model, excluding the input layer. 
                       All layers should have a parent class of LinearLayer.
    """
    def __init__(self, learning_rate:float, modules:List[LinearLayer]):
        self.learning_rate = learning_rate
        self.modules = modules

    def step(self):
        """
        Perform a single optimization step, updating the parameters of all layers in 'modules'.
        """
        for module in self.modules:
            module.W -= self.learning_rate * module.G.mean(axis=0)
            pass
        pass
    pass

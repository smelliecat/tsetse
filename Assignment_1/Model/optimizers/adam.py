import numpy as np
from typing import List
from layers.linear import LinearLayer

class AdamSolver:
    """
    Implements the Adam optimization algorithm.
    Adam Algorithm:
    Adam combines the benefits of both AdaGrad and RMSProp.
    W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    where m_hat = m / (1 - beta1^t) and v_hat = v / (1 - beta2^t)

    Parameters:
    lr (float): Learning rate
    modules (List[LinearLayer]): List of layers in the model (excluding the input layer)
    beta1 (float, optional): Exponential decay rate for first moment estimate, default is 0.9
    beta2 (float, optional): Exponential decay rate for second moment estimate, default is 0.999
    epsilon (float, optional): Small constant to prevent division by zero, default is 1e-8
    """

    def __init__(self, learning_rate:float, modules: List[LinearLayer], beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-8):
       self.learning_rate = learning_rate
        self.modules = modules
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        for module in self.modules:
            module.m = 0
            module.v = 0
       

    def step(self):
        """
        """
        self.t += 1
        for module in self.modules:
            module.m = self.beta1 * module.m + (1 - self.beta1) * module.G.mean(axis=0)
            module.v = self.beta2 * module.v + (1 - self.beta2) * module.G.mean(axis=0)**2

            m_m_hat = module.m / (1 - self.beta1**self.t)
            m_v_hat = module.v / (1 - self.beta2**self.t)

            module.W -= self.lr * m_m_hat / (np.sqrt(m_v_hat) + self.epsilon)
            pass
        pass
    pass



 


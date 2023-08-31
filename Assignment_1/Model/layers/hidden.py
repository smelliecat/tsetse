import numpy as np

from layers.linear import LinearLayer

from activation.relu import ReLU
from activation.sigmoid import Sigmoid
from activation.tanh import Tanh

class HiddenLayer(LinearLayer):
    def __init__(self, input_dimension, output_dimension, activation='ReLU') -> None:
        super().__init__(input_dimension, output_dimension)

        if activation == 'Sigmoid':
            self.activation = Sigmoid(self)

        elif activation == 'Tahn':
            self.activation = Tanh(self)

        else:
            self.activation = ReLU(self)

    def forward(self):
        _ = super().forward()
        self.activated_output = self.activation.forward()
        return self.activated_output
    
    def backward(self, downstream):
        activation_grad = self.activation.backward(downstream)
        super().backward(activation_grad)
        
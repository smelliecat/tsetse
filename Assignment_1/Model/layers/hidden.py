import numpy as np

from Model.layers.linear import LinearLayer

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
            self.activation = ReLU()

    def forward(self):
        _out = super().forward()
        self.activated_output = self.activation.forward(_out)
        return _out
    
    def backward(self, downstream):
        print("Shape of downstream in HiddenLayer:", downstream.shape)
        activation_grad = self.activation.backward(downstream)
        super().backward(activation_grad)
        
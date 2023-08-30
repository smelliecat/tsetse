import numpy as np

class Sigmoid:
    """
    Implements the Sigmoid activation function.
    
    The sigmoid function is defined as: f(x) = 1 / (1 + e^{-x})
    
    Attributes:
        input_layer: The layer that provides the input to this activation function.
        
    Methods:
        forward(): Applies the Sigmoid activation function to the output of the input layer.
        backward(downstream): Computes the gradient of the loss with respect to the input, which is then passed back to the previous layers.
    """

    def __init__(self, input_layer) -> None:
        self.input_layer = input_layer
    
    def forward(self):
        self.input_array = self.input_layer.forward()
        sigmoid = 1 / (1 + np.exp(-self.input_array))
        return sigmoid

    def backward(self, downstream):
        input_grad = downstream * np.exp(-np.maximum(self.input_array, 0)) / (
                    (1 + np.exp(-self.input_array)) * (1 + np.exp(-np.abs(self.input_array))))
        self.input_layer.backward(input_grad)
    
    pass



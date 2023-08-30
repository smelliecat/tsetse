import numpy as np

class Tanh:
    """
    Implements the hyperbolic tangent (tanh) activation function.
    
    The tanh function is defined as: f(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    
    Attributes:
        in_layer: The layer that provides the input to this activation function.
        
    Methods:
        forward(): Applies the tanh activation function to the output of the input layer.
        backward(downstream): Computes the gradient of the loss with respect to the input, 
                           which is then passed back to the previous layers.
    """

    def __init__(self, input_layer) -> None:
        self.input_layer = input_layer
    
    def forward(self):
        self.input_array = self.input_layer.forward()
        self.tahn_output = (np.exp(self.input_array) - (np.exp(-self.input_array)) / np.exp(self.input_array) + (np.exp(-self.input_array)))
        return self.tahn_output

    def backward(self, downstream):
        tanh_grad = 1 - self.tahn_output**2
        input_grad = downstream * tanh_grad
        self.input_layer.backward(input_grad)



# They can NOT use the np.tahn function


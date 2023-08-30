import numpy as np
class ReLU:
    """
    Implements the Rectified Linear Unit (ReLU) activation function.
    
    The relu function is defined as: f(x) = max(0,x)

    Attributes:
        input_layer: The layer that feeds input into this activation function.
        input_dimension: The shape of the output from the input_layer.
        output_dimension: The shape of the output of this layer, which 
        is the same as the input_dimension for ReLU.
        
    Methods:
        forward(): Applies the ReLU activation function to the output 
        of the input layer.
        backward(downstream): Computes the gradient of the loss with 
        respect to the input, to be passed back to the previous layers.
    """

    def __init__(self, input_layer) -> None:
        self.input_layer = input_layer
        self.input_dimension = input_layer.output_dimension
        self.output_dimension = self.input_dimension

    def forward(self):
        self.input_array = self.input_dimension.forward()
        self.output_array = np.maximum(self.input_array, 0)
        return self.output_array

    def backward(self, downstream):
        input_grad = downstream * (self.input_array > 0)
        self.input_layer.backward(input_grad)

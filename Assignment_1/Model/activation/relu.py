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

    # def __init__(self, input_layer) -> None:
    #     self.input_layer = input_layer
    #     self.input_dimension = input_layer.output_dimension
    #     self.output_dimension = self.input_dimension
    @staticmethod
    def forward(input_array):
        # self.input_array = self.input_layer.forward()
        output_array = np.maximum(input_array, 0)
        return output_array
    
    @staticmethod
    def backward(downstream):
        print("IN RELUUUU ", downstream.shape)
        input_grad = downstream * (downstream > 0)
        # self.input_layer.backward(input_grad)
        return input_grad

import numpy as np
# from linear import LinearLayer

class InputLayer:
    """
    Represents the input layer of a neural network. The input layer essentially acts as
    a pass-through for the input data and does not perform any transformations.

    Attributes:
        W: Identity matrix corresponding to the dimensions of the input data. 
           This is included for compatibility with other layers but is not used.
        output_dimension: The dimensions of the output (same as input in the case of InputLayer).

    Methods:
        forward(input_data): Passes the input data through the layer without altering it.
        backward(downstream): Placeholder function; no actual backward pass computations 
                              are performed in the input layer.
    """
     
    def __init__(self, input_dimension) -> None:
        self.W = np.eye(input_dimension)
        self.output_dimension = np.array[None, input_dimension]
    
    def forward(self, input_data):
        """
        """
        self.input_array = input_data
        self.output_array[0] = self.input_array.shape[0]

        return self.input_array

    def backward(self, downstream):
        """
        """
        pass

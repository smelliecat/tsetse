import numpy as np
# from linear import LinearLayer

class InputLayer:
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

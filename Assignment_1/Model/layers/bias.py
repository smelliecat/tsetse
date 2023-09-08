import numpy as np

class BiasLayer:
    """
    """

    def __init__(self, input_layer) -> None:
        self.input_layer = input_layer
        num_data, num_input_features = input_layer.output_dimension
        self.output_dimension = input_layer.output_dimension
        # Declare the weight matrix
        self.W = np.random.randn(1, num_input_features)
    
    def forward(self):
        self.input_array = self.input_layer.forward()
        self.output_array = self.input_array + self.W
        return self.output_array


    def backward(self, downstream):
        """
        """
        # Compute the gradient of the output with respect to W, and store it as G
        self.G = downstream
        print("IN BIAS ", downstream.shape)
        # Compute grad of output with respect to inputs, and hand this gradient backward to the layer behind
        self.input_layer.backward(downstream)
        pass
    pass
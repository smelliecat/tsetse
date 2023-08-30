import numpy as np

class LinearLayer:
    """
    A class representing a fully connected (linear) layer in a neural network.
    
    Methods:
        forward(): Computes the forward pass of the layer.
        backward(dwnstrm): Computes the backward pass, propagating the gradient.
    """

    def __init__(self, input_layer, output_dimension) -> None:
        """
        Initialize the layer.

        Parameters:
            input_layer: The preceding layer in the neural network.
            output_dimension: Number of neurons in the current layer.
        
        Raises:
            AssertionError: If input layer dimensions are not a list of 1D linear feature data.
        """
        assert len(input_layer.out_dims) == 2, "Input layer must contain a list of 1D linear feature data."
        self.input_layer = input_layer
        num_data, num_in_features = input_layer.out_dims
        self.out_dims = np.array([num_data, output_dimension])
        # Declare the weight matrix and initialize it
        self.W = np.random.randn(num_in_features, output_dimension) / np.sqrt(num_in_features)


    def forward(self):
        """
        Compute the forward pass for the layer, i.e., compute XW.
        """
        self.input_array = self.input_layer.forward()
        self.output_array = self.input_array @ self.W
        return self.output_array

    def backward(self, downstream):
        """
        Compute the backward pass for the layer, propagating the gradient backward.
        """
        # Compute gradient with respect to weights
        self.G = self.in_array[:, :, np.newaxis] * downstream[:, np.newaxis]
        # Compute gradient with respect to inputs
        input_grad = (self.W @ downstream[:, :, np.newaxis]).squeeze(axis=-1)
        self.in_layer.backward(input_grad)

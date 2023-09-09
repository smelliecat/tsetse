import numpy as np

class BiasLayer:
    """
    This layer adds a bias term to the output of the input layer. 
    Each feature dimension of the input gets a bias term.
    """

    def __init__(self, input_layer) -> None:
        self.input_layer = input_layer
        num_data, num_input_features = input_layer.output_dimension
        self.output_dimension = input_layer.output_dimension
        # Declare the weight matrix
        # TODO: Declare the weight matrix (bias term) for the layer. Replace `None` with appropriate code.
        self.W = None 
    
    def forward(self):
        # self.input_array = self.input_layer.forward()
        # self.output_array = self.input_array + self.W
        # return self.output_array

        """
        Perform the forward pass through the bias layer.
        
        Returns:
        The output array after adding the bias terms.
        """
        # TODO: Get the output from the input layer and store it in `self.input_array`. Replace `None` with appropriate code.
        self.input_array = None 
        
        # TODO: Perform the actual forward computation and store the result in `self.output_array`. Replace `None` with appropriate code.
        self.output_array = None 


    def backward(self, downstream):
        """
        Perform the backward pass.
        
        Parameters:
        - downstream: The gradient of the loss function with respect to the output of this layer.
        """
        # TODO: Compute the gradient of the output with respect to the bias term `self.W` and store it in `self.G`. Replace `None` with appropriate code.
        self.G = None
        
        # TODO: Compute the gradient of the output with respect to the inputs and pass this backward to the layer behind. Replace `None` with appropriate code.
        self.input_layer.backward(None)
    

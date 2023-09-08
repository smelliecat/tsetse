import numpy as np

class SquareLoss:
    """
    """

    def __init__(self, input_dimension, labels=None) -> None:
        self.input_layer = input_dimension
        self.labels = labels
        self.labels = self.labels.reshape(-1, 1)
    
    def set_labels(self,labels):
        self.labels = labels


    def forward(self):
        """Loss value is (1/2M) || X-Y ||^2"""
       

        self.in_array = self.input_layer.forward()
        # print("LOSSS num data ISSS: ", self.in_array.shape)
        self.num_data = self.in_array.shape[1]
        # print("LOSSS num data ISSS: ", self.num_data)
        self.out_array = (0.5 / self.num_data) * np.linalg.norm(self.in_array - self.labels) ** 2
        # print(self.out_array.shape)
        print("Shape of self.in_array:", self.in_array.shape)
        print("Shape of self.labels:", self.labels.shape)
        return self.out_array

    def backward(self):
        """
        Gradient is (1/M) (X-Y), where N is the number of training samples
        """
        print(self.labels.shape)
        self.pass_back = (self.in_array - self.labels) / self.num_data
        print("Shape of self.pass_back:", self.pass_back.shape)
        self.input_layer.backward(self.pass_back)  # hand the gradient of loss with respect to inputs back to previous layer
        pass
    pass
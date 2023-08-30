import numpy as np

from layers.linear import LinearLayer

from activation.relu import ReLU
from activation.sigmoid import Sigmoid
from activation.tahn import Tahn

class HiddenLayer(LinearLayer):
    def __init__(self, input_dimension, output_dimension) -> None:
        super().__init__(input_dimension, output_dimension)
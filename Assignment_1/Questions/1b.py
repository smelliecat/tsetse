from Model.layers.network import BaseNetwork
import numpy as np
import matplotlib.pyplot as plt
from Model.layers.input import InputLayer
from Model.layers.hidden import HiddenLayer
from Model.loss.square_loss import SquareLoss
from Model.layers.bias import BiasLayer
from Model.layers.output_layer import OutputLayer
from Model.optimizers.sgd import SGDSolver
from Model.optimizers.adam import AdamSolver
from Data.data import Data
from Data.generator import q1
from Model.evaluate.evaluate import evaluate_model



Number_of_iterations = 10000
Step_size = 0.5

class Network(layers.BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer):
        # you should always call __init__ first
        super().__init__()
        # TODO: define your network architecture here
        self.linear = layers.Linear(data_layer, num_out_features=2)
        self.bias = layers.Bias(self.linear)
        # For prob 3 and 4:
        # layers.ModuleList can be used to add arbitrary number of layers to the network
        # e.g.:
        # self.MY_MODULE_LIST = layers.ModuleList()
        # for i in range(N):
        #     self.MY_MODULE_LIST.append(layers.Linear(...))

        # TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.bias)
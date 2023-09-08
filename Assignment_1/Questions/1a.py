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


Number_of_iterations = 300
Step_size = 0.01

class Network(BaseNetwork):
    def __init__(self, data_layer):
        super().__init__()
        data = data_layer.forward()
        self.input_layer = InputLayer(data_layer)
        print("data shape in network", data.shape)
        self.hidden_layer1 = HiddenLayer(self.input_layer, 1)
        self.bias_layer1 = BiasLayer(self.hidden_layer1)
        self.output_layer1 = OutputLayer(self.bias_layer1, 1)
        self.set_output_layer(self.output_layer1)




class Trainer:
    def __init__(self) -> None:
        pass

    def define_net(self, data_layer, parameters=None):
        network = Network(data_layer)
        return network
    
    def net_setup(self, train_data):
        features, labels = train_data
        self.data_layer = Data(features)
        print("data out dim in ney_setup", self.data_layer.output_dimension)
        self.network = self.define_net(self.data_layer)
        self.loss_layer = SquareLoss(self.network.get_output_layer(), labels=labels)
        # self.optimizer = SGDSolver(Step_size, self.network.get_modules_with_parameters())
        self.optimizer = AdamSolver(Step_size, self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optimizer
    
    def train_step(self):
        loss = self.loss_layer.forward()
        self.loss_layer.backward()
        self.optimizer.step()
        # print(loss)
        return loss
    
    def train(self, number_of_iterations):
        train_losses = []

        for _ in range(number_of_iterations):
            train_losses.append(self.train_step())

        return train_losses
    

def main(test=False):
    # setup the trainer
    trainer = Trainer()

    # DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        data = q1()
        data_layer, network, loss_layer, optimizer = trainer.net_setup(data['train'])
        loss = trainer.train(Number_of_iterations)
        plt.plot(loss)
        plt.ylabel('Loss of NN')
        plt.xlabel('Number of Iterations')
        plt.show()

        # Now let's use the test data
        x_test, y_test = data['test']
        test_data_layer = Data(x_test)
        network.input_layer = InputLayer(test_data_layer)
        network.hidden_layer1.input_layer = network.input_layer

        # Get predictions for test data
        y_pred = network.output_layer1.forward()

        metrics = evaluate_model(y_test, y_pred)
        # Print the metrics for review
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Plot actual vs predicted on test data
        plt.figure(figsize=(10, 6))
        plt.scatter(x_test, y_test, label='Actual', alpha=0.6)
        plt.scatter(x_test, y_pred, label='Predicted', alpha=0.6)
        plt.xlabel('Feature (x)')
        plt.ylabel('Target (y)')
        plt.title('Test Data and Model Predictions')
        plt.legend()
        plt.show()

    else:
        # DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out


if __name__ == "__main__":
    main()
    pass

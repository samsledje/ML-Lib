"""
Neural Networks
"""
import numpy as np
from utils import *

class FeedForwardNet:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):

        # Architecture
        self.input_n = input_neurons
        self.hidden_n = hidden_neurons
        self.output_n = output_neurons

        self.lr = 0.1

        # Weights / Biases
        self.hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
        self.hidden_biases = np.random.uniform(size=(1, hidden_neurons))
        self.output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
        self.output_biases = np.random.uniform(size=(1, output_neurons))

    def train(self, X, y, epochs=5000):
        for i in range(epochs):
            # Forward Step
            hidden_activations = _sigmoid(np.dot(X, self.hidden_weights) + self.hidden_biases)
            output_activations = _sigmoid(np.dot(hidden_activations, self.output_weights) + self.output_biases)

            # Back Propagation
            errors = y - output_activations
            output_deriv = _dev_sigmoid(output_activations)
            hidden_deriv = _dev_sigmoid(hidden_activations)

            del_output = errors * output_deriv
            hidden_error = del_output.dot(self.output_weights.T)
            del_hidden = hidden_error * hidden_deriv

            self.output_weights = self.output_weights + hidden_activations.T.dot(del_output) * self.lr
            self.output_biases = self.output_biases + np.sum(del_hidden, axis=0) * self.lr

            self.hidden_weights = self.hidden_weights + X.T.dot(del_hidden) * self.lr
            self.hidden_biases = self.hidden_biases + np.sum(del_output, axis=0) * self.lr

    def predict(self, X):
        pass

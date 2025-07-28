import math
from .activation import relu, sigmoid, tanh

class Node:
    def __init__(self, weights, bias=0.0, activation='relu'):
        self.weights = weights  # list of weights
        self.bias = bias
        self.activation = activation
        self.last_input = []
        self.last_output = None

    def activate(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Input size must match weight size")

        self.last_input = inputs

        # Weighted sum
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

        # Apply activation
        self.last_output = self._apply_activation(z)
        return self.last_output

    def _apply_activation(self, z):
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'tanh':
            return tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def __repr__(self):
        return f"Node(weights={self.weights}, bias={self.bias}, activation='{self.activation}')"

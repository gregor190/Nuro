from .node import Node
from .activation import relu_derivative, sigmoid_derivative, tanh_derivative

class Layer:
    def __init__(self, size, input_size, activation='relu'):
        """Create a layer with `size` nodes, each expecting `input_size` inputs."""
        self.activation = activation
        self.nodes = [
            Node(weights=[0.0] * input_size, bias=0.0, activation=activation)
            for _ in range(size)
        ]
        self.last_inputs = []
        self.last_outputs = []

    def set_weights(self, weights_list):
        """Set weights and biases for each node.

        weights_list should be a list of tuples: (weights, bias)
        """
        if len(weights_list) != len(self.nodes):
            raise ValueError("Mismatch between weights and number of nodes")

        for node, (weights, bias) in zip(self.nodes, weights_list):
            node.weights = weights
            node.bias = bias

    def forward(self, inputs, dropout_rate=0.0):
        """Run inputs through all nodes and return their outputs."""
        import random
        self.last_inputs = inputs
        self.last_outputs = []
        for node in self.nodes:
            if random.random() > dropout_rate:
                output = node.activate(inputs)
            else:
                output = 0.0  # Dropout
            self.last_outputs.append(output)
        return self.last_outputs

    def backward(self, output_errors, learning_rate):
        """Backpropagate errors and update weights."""
        for i, node in enumerate(self.nodes):
            error = output_errors[i]
            z = sum(w * x for w, x in zip(node.weights, self.last_inputs)) + node.bias

            # Choose derivative
            if node.activation == 'relu':
                derivative = relu_derivative(z)
            elif node.activation == 'sigmoid':
                derivative = sigmoid_derivative(z)
            elif node.activation == 'tanh':
                derivative = tanh_derivative(z)
            else:
                raise ValueError(f"Unknown activation: {node.activation}")

            delta = error * derivative

            # Update weights and bias
            for j in range(len(node.weights)):
                node.weights[j] -= learning_rate * delta * self.last_inputs[j]
            node.bias -= learning_rate * delta

    def debug(self):
        for i, node in enumerate(self.nodes):
            print(f"Node {i}: weights={node.weights}, bias={node.bias}, activation={node.activation}")

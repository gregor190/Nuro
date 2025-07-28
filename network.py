from .layers import Layer

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, size, input_size=None, activation='relu'):
        """Add a layer to the network. First layer must specify input_size."""
        if not self.layers and input_size is None:
            raise ValueError("First layer must specify input_size")
        if input_size is None:
            input_size = len(self.layers[-1].nodes)
        layer = Layer(size=size, input_size=input_size, activation=activation)
        self.layers.append(layer)

    def set_weights(self, weights_list):
        """Set weights for all layers. weights_list is a list of layer-wise weights."""
        if len(weights_list) != len(self.layers):
            raise ValueError("Mismatch between weights and layers")
        for layer, layer_weights in zip(self.layers, weights_list):
            layer.set_weights(layer_weights)

    def forward(self, inputs):
        """Run inputs through all layers."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, expected_output, learning_rate):
        """Backpropagate error from output layer to input layer."""
        output = self.layers[-1].last_outputs
        errors = [o - e for o, e in zip(output, expected_output)]

        for layer in reversed(self.layers):
            layer.backward(errors, learning_rate)
            new_errors = []
            for i in range(len(layer.nodes[0].weights)):
                error_sum = sum(
                    node.weights[i] * errors[j]
                    for j, node in enumerate(layer.nodes)
                )
                new_errors.append(error_sum)
            errors = new_errors

    def train(self, inputs, expected_output, learning_rate=0.1):
        """Train the network on one input-output pair."""
        self.forward(inputs)
        self.backward(expected_output, learning_rate)

    def debug(self):
        print("🧠 NURO Network Debug:")
        for i, layer in enumerate(self.layers):
            print(f" Layer {i}:")
            layer.debug()

    def summary(self):
        print("📊 NURO Network Summary:")
        for i, layer in enumerate(self.layers):
            print(f" Layer {i}: {len(layer.nodes)} nodes, activation='{layer.activation}'")

import math

class Neuron:
    def __init__(self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.activation_functions = {
            "_relu": self._relu,
            "_sigmoid": self._sigmoid,
            "_tanh": self._tanh
        }
        if func not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {func}")
        self.func = func

    def predict(self, input_data):
        if len(input_data) != len(self.weights):
            raise ValueError("Input data and weights must have the same length.")

        weighted_sum = sum(w * x for w, x in zip(self.weights, input_data))
        weighted_sum += self.bias

        return self.activate(weighted_sum)

    def activate(self, weighted_sum):
        return self.activation_functions[self.func](weighted_sum)

    @staticmethod
    def _relu(x):
        return max(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def _tanh(x):
        return math.tanh(x)

    def change_weights(self, new_weights):
        if len(new_weights) != len(self.weights):
            raise ValueError("New weights must have the same length as the current weights.")
        self.weights = new_weights

    def change_bias(self, new_bias):
        self.bias = new_bias
import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha*x)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def d_sigmoid(x):
        sx = Activation.sigmoid(x)
        return sx * (1 - sx)
    
    @staticmethod
    def d_tanh(x):
        return 1.0 - np.tanh(x)**2

    @staticmethod
    def d_relu(x):
        return (x > 0) * 1
    
    @staticmethod
    def d_leaky_relu(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

class Layer_dense:
    activations = {
        "Sigmoid": (Activation.sigmoid, Activation.d_sigmoid),
        "ReLU": (Activation.relu, Activation.d_relu),
        "Tanh": (Activation.tanh, Activation.d_tanh),
        "LeakyReLU": (Activation.leaky_relu, Activation.d_leaky_relu)
    }

    def __init__(self, number_inputs, number_outputs, activation_function):
        self.weights = 0.01 * np.random.randn(number_inputs, number_outputs)  
        self.bias = 0.01 * np.random.randn(1, number_outputs)

        if activation_function in Layer_dense.activations:
            self.activation, self.activation_deriv = Layer_dense.activations[activation_function]
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def Forward(self, batch_inputs):
        self.inputs = batch_inputs # Note these can be the inputs to the network or from the previous layer
        self.weighted_sum = np.dot(batch_inputs, self.weights) + self.bias
        self.outputs = self.activation(self.weighted_sum)
        return self.outputs

    def BackProp(self, dvalues):
        self.ds_activation = self.activation_deriv(self.weighted_sum)
        self.omega_layer = dvalues * self.ds_activation
        self.weight_gradient = np.dot(self.inputs.T, self.omega_layer)
        self.bias_gradient = np.sum(self.omega_layer, axis=0, keepdims=True)
        dinputs = np.dot(self.omega_layer, self.weights.T)

        return dinputs
    
    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient

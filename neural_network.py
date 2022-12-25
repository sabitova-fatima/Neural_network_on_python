import numpy as np

# the initial class we will inherit from
class Layer:
	def __init__(self):
		self.input = None
		self.output = None
	def forward(self, input):
		pass
	def backward(self, output_gradient, learning_rate):
		pass

# Dense layer
# Y = W * X * B
# Y - output matrix (column vector of size i) - output_size
# W - weights matrix (j * i) - output_size * input_size
# X - input matrix (column vector of size j) - input_size
# B - bias (column vector of size j)

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

# activation is just another layer that takes input and gives output
class Activation(Layer):
	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	def forward(self, input):
		self.input = input
		result = self.activation(self.input)
		return result

	def backward(self, output_gradient, learning_rate):
		result = np.multiply(output_gradient, self.activation_prime(self.input))
		return result

# Tanh activation function
class Tanh(Activation):
	def __init__(self):
		def tanh(x):
			return np.tanh(x)

		def tanh_prime(x):
			return 1 - np.tanh(x) ** 2

		super().__init__(tanh, tanh_prime)


# MSE loss function
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, show_error_state = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if show_error_state:
            print(f"{e + 1}/{epochs}, error={error}")
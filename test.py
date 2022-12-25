import numpy as np
from neural_network import Dense, Tanh, mse, mse_prime, predict, train

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
	Dense(2, 3),
	Tanh(),
	Dense(3, 1),
	Tanh()
]

train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

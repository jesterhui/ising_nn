import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state

np.random.seed()

X, y = fetch_openml('mnist_784', return_X_y=True)
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
y = y.astype('float')
y = y.reshape((y.size, 1))
X = X.reshape((X.shape[0], -1))
X[X>=0.5] = 1
X[X<0.5] = 0


class RestrictedBoltzmannMachine:

    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.momentum = 0
        self.weights = None
        self.alpha = 1e-1

    def forward_pass(self, x):
        return self.sigmoid(x @ self.weights) > np.random

    def contrastive_divergence(self, x, y, training_iter=1000):
        x = np.insert(x, 0, 1, axis=1)
        x = x[:10, :].reshape(10, 785)

        self.weights = np.random.rand(self.hidden_layers, x.shape[1])
        print(self.weights)
        for _ in range(training_iter):
            ind = np.random.randint(10, size=2)
            x_train = x[ind, :].reshape(2, 785)
            old_weights = self.weights.copy()
            hidden_activations = self.sigmoid(self.weights @ x_train.T)
            hidden_activations[0, :] = 1
            hidden_values = np.greater_equal(hidden_activations, np.random.rand(hidden_activations.shape[0], hidden_activations.shape[1]))
            visible_activations = self.sigmoid(self.weights.T @ hidden_values)
            visible_values = np.greater_equal(visible_activations, np.random.rand(visible_activations.shape[0], visible_activations.shape[1]))
            fig, ax = plt.subplots(1, 3)
            ax.ravel()[0].imshow(x_train[0, 1:].reshape((28, 28)))
            ax.ravel()[1].imshow(visible_activations[1:, 0].reshape((28, 28)))
            ax.ravel()[2].imshow(visible_values[1:, 0].reshape((28, 28)))
            plt.show()
            print(hidden_activations, y.shape, visible_activations)
            #
            update = self.momentum * 0.5 + self.alpha * (hidden_activations @ x_train - hidden_activations @ visible_activations.T)
            self.weights += update
            self.momentum = update
            print(np.mean((visible_activations.T - x_train)**2))
            print(self.weights)
            if np.sum(np.abs(self.weights-old_weights)) < 1e-3:
                break

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


R = RestrictedBoltzmannMachine(10)
R.contrastive_divergence(X, y)

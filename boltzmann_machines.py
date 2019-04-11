import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state

mpl.rc('font', **{'family': 'serif'})
mpl.rc('text', usetex=True)


class RestrictedBoltzmannMachine:
    """Restricted Boltzmann Machine.

    Attributes:
        weights (obj): NumPy array of dimension (h+1, d+1) where h is the
        number of hidden layers and d is the number of input features. The +1
        for each is due to the bias term.
        alpha (float): Learning rate, which specifies the step size for
        the contrastive divergence training.
        eta (float): Value specifying the strength of the momentum term for the
        contrastive_divergence.

    """

    def __init__(self, hidden_layers, alpha=1e-1, eta=0.5):
        """Pass network, training parameters to class.

        Args:
            hidden_layers (int): Number of nodes in the hidden layer.
            alpha (float): Learning rate, which specifies the step size for
            the contrastive divergence training.
            eta (float): Value specifying the strength of the momentum term for
            the contrastive_divergence.

        """
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.eta = eta
        self.weights = None

    def forward_pass(self, visible_values):
        """Pass visible node values forward through network to hidden layers.

        Args:
            visible_values (obj): NumPy array containing visible node values.

        Returns:
            obj: NumPy array containing activations for each node in the hidden
            layer.
            obj: NumPy array containing sampled values for each node in the
            hidden layer.

        """
        hidden_activations = self.sigmoid(self.weights @ visible_values)
        hidden_activations[0, :] = 1
        h_i, h_j = hidden_activations.shape
        hidden_values = hidden_activations > np.random.rand(h_i, h_j)
        return hidden_activations, hidden_values

    def backward_pass(self, hidden_values):
        """Pass hidden node values backwards through network to visible layers.

        Args:
            hidden_values (obj): NumPy array containing hidden node values.

        Returns:
            obj: NumPy array containing activations for each node in the
            visible layer.
            obj: NumPy array containing sampled values for each node in the
            visible layer.

        """
        visible_activations = self.sigmoid(self.weights.T @ hidden_values)
        visible_activations[0, :] = 1
        v_i, v_j = visible_activations.shape
        visible_values = visible_activations > np.random.rand(v_i, v_j)
        return visible_activations, visible_values

    def gibbs(self, visible_values):
        """Perform one Gibbs sampling step to generate new visible nodes.

        Args:
            visible_values (obj): NumPy array containing visible node values.

        Returns:
            obj: NumPy array containing new visible node activations
            obj: NumPy array containing new visible node values.

        """
        _, hidden_values = self.forward_pass(visible_values)
        visible_activations, visible_values = self.backward_pass(hidden_values)
        return visible_activations, visible_values

    def contrastive_divergence(self, x, k=10, epochs=100, batch_size=100):
        """Train restricted boltzmann machine using contrastive divergence.

        Args:
            x (obj): NumPy array of dimension (n, d) containing data.
            epochs (int): Number of training epochs.

        """
        momentum = 0  # initialize momentum as 0
        x = np.insert(x, 0, 1, axis=1)  # insert ones row to use as bias unit
        n_batches = int(x.shape[0] / batch_size)  # calculate number of batches
        # initialize weights randomly
        self.weights = np.random.rand(self.hidden_layers+1, x.shape[1])
        # initialize bias units to have weight 0
        self.weights[0, :] = 0
        self.weights[:, 0] = 0

        # begin training loop
        for epoch in range(epochs):
            print(epoch)

            np.random.shuffle(x)
            for i in range(n_batches):
                # slice training batch
                x_train = x[i*batch_size:i*batch_size+batch_size, :]
                # use more aggressive training parameters later on in training
                if epoch == 15:
                    self.alpha = 1e-1
                    self.eta = 0.9
                    momentum = 0
                # perform first forward pass
                (hidden_activations,
                 hidden_values) = self.forward_pass(x_train.T)
                # calulcate positive phase
                positive_phase = hidden_activations @ x_train
                _, visible_values = self.backward_pass(hidden_values)
                # perorm gibbs sampling
                for __ in range(k - 1):
                    (visible_activations,
                     visible_values) = self.gibbs(visible_values)

                (hidden_activations,
                 hidden_values) = self.forward_pass(visible_values)
                negative_phase = hidden_activations @ visible_values.T
                update = ((self.alpha * (positive_phase - negative_phase)
                           / batch_size) + self.eta * momentum)
                self.weights += update
                momentum = update

            # monitor training progress
            if (epoch + 1) % 5 == 0:
                _, visible_values = self.gibbs(x.T)

                for ___ in range(k - 1):
                    (visible_activations,
                     visible_values) = self.gibbs(visible_values)
                fig, ax = plt.subplots(1, 2)
                ax.ravel()[0].imshow(x[0, 1:].reshape((28, 28)), cmap='Greys')
                ax.ravel()[0].set_title('Original Digit')
                ax.ravel()[1].imshow(visible_values[1:, 0]
                                     .reshape((28, 28)), cmap='Greys')
                ax.ravel()[1].set_title('Generated Digit')
                for i in range(2):
                    ax.ravel()[i].set_xticks([])
                    ax.ravel()[i].set_yticks([])
                plt.tight_layout()
                plt.savefig('images/visualize_{}epochs.png'.format(epoch
                                                                   + 1),
                            dpi=400, bbox_inches='tight')
                plt.close()
                error = self.reconstruction_error(x, visible_activations)
                print(error)

    @staticmethod
    def sigmoid(z):
        """Sigmoid utility.

        Args:
            z (obj/int): Input for sigmoid function.

        Returns:
            obj: Output of sigmoid function.

        """
        sigma = 1 / (1 + np.exp(-z))
        return sigma

    @staticmethod
    def reconstruction_error(x, visible_activations):
        """Calculate reconstruction_error.

        Args:
            x (obj): NumPy array of dimension (n, d) containing data.
            visible_activations (type): NumPy array of dimension (d, n)
            containing visible activations.

        Returns:
            float: Average reconstruction error for each image.

        """
        reconstruction_error = (x.shape[1]
                                * np.mean((visible_activations-x.T)**2))
        return reconstruction_error


np.random.seed()

X, y = fetch_openml('mnist_784', return_X_y=True)
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
y = y.astype('float')
y = y.reshape((y.size, 1))
X = X.reshape((X.shape[0], -1))
X[X >= 0.5] = 1
X[X < 0.5] = 0
X = X[:10000, :]
X = X.reshape((X.shape[0], -1))

R = RestrictedBoltzmannMachine(144)
R.contrastive_divergence(X)
fig, ax = plt.subplots(12, 12)
for i in range(144):
    ax.ravel()[i].imshow(R.weights[i, 1:].reshape(28, 28), cmap='coolwarm')
    ax.ravel()[i].set_xticks([])
    ax.ravel()[i].set_yticks([])
plt.tight_layout()
plt.savefig('images/latent_space.png', dpi=400, bbox_inches='tight')

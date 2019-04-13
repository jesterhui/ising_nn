"""
Generative models used to generate Ising model lattice configurations for
CHE696 final project.
"""
import matplotlib as mpl
import numpy as np

mpl.rc('font', **{'family': 'serif'})
mpl.rc('text', usetex=True)


class RestrictedBoltzmannMachine:
    """Restricted Boltzmann machine.

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
        if visible_values.shape[0] == self.weights.shape[1] - 1:
            visible_values = np.insert(visible_values, 0, 1, axis=0)
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

    def contrastive_divergence(self, x, k=10, epochs=100, batch_size=100,
                               dbn_bias=True):
        """Train restricted boltzmann machine using contrastive divergence.

        Args:
            x (obj): NumPy array of dimension (n, d) containing data.
            k (int): Number of Gibbs sampling steps used during training.
            Emperically 10 is usually good.
            epochs (int): Number of training epochs.
            batch_size (int): Size of training batch.
            dbn_bias (bool): specifies whether to include bias for later layers
            of DBN.

        """
        momentum = 0  # initialize momentum as 0
        if dbn_bias is True:
            x = np.insert(x, 0, 1, axis=1)  # insert ones row for bias unit
            # initialize weights randomly
            self.weights = np.random.rand(self.hidden_layers+1, x.shape[1])
        else:
            # initialize weights randomly
            self.weights = np.random.rand(self.hidden_layers, x.shape[1])

        n_batches = int(x.shape[0] / batch_size)  # calculate number of batches
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


class DeepBeliefNetwork:
    """Deep belief network.

    Attributes:
        weights (obj): NumPy array of dimension (h+1, d+1) where h is the
        number of hidden layers and d is the number of input features. The +1
        for each is due to the bias term.
        alpha (float): Learning rate, which specifies the step size for
        the contrastive divergence training.
        eta (float): Value specifying the strength of the momentum term for the
        contrastive_divergence.
        hidden_layers_spec (list): List specifying the number of nodes in
        each hidden layer, in the order of visible -> associative memory.
        layers (list): List of RestrictedBoltzmannMachine objects used for each
        layer.

    """

    def __init__(self, hidden_layers_spec, alpha=1e-1, eta=0.5):
        """Pass network, training parameters to class.

        Args:
            hidden_layers_spec (list): List specifying the number of nodes in
            each hidden layer, in the order of visible -> associative memory.
            alpha (float): Learning rate, which specifies the step size for
            the contrastive divergence training.
            eta (float): Value specifying the strength of the momentum term for
            the contrastive_divergence.

        """
        self.hidden_layers_spec = hidden_layers_spec
        self.alpha = alpha
        self.eta = eta
        self.weights = None
        self.layers = []

    def forward_pass(self, visible_values, n_associative=20):
        """Pass visible node values forward through network to hidden layers,
         and then pass back and forth in the associative memory.

        Args:
            visible_values (obj): NumPy array containing visible node values.
            n_associative (int): Number of passes to perform in associative
            memory.

        Returns:
            obj: NumPy array containing sampled values for each node in the
            hidden layer.

        """
        n_l = len(self.hidden_layers_spec)
        _, hidden_values = self.layers[0].forward_pass(visible_values)
        for i in range(1, n_l-1):
            _, hidden_values = self.layers[i].forward_pass(visible_values)
        for __ in range(n_associative):
            _, hidden_values = self.layers[n_l-1].gibbs(hidden_values)
        _, hidden_values = self.layers[n_l-1].forward_pass(hidden_values)
        return hidden_values

    def backward_pass(self, hidden_values, n_associative=20):
        """Pass hidden node values backwards through network to visible layers.

        Args:
            hidden_values (obj): NumPy array containing hidden node values.
            n_associative (int): Number of passes to perform in associative
            memory.

        Returns:
            obj: NumPy array containing sampled values for each node in the
            visible layer.

        """
        n_l = len(self.hidden_layers_spec)
        for __ in range(n_associative):
            _, hidden_values = self.layers[n_l-1].backward_pass(hidden_values)
            _, hidden_values = self.layers[n_l-1].forward_pass(hidden_values)
        _, hidden_values = self.layers[n_l-1].backward_pass(hidden_values)
        for i in reversed(range(1, n_l-1)):
            _, hidden_values = self.layers[i].backward_pass(hidden_values)
        _, visible_values = self.layers[0].backward_pass(hidden_values)
        return visible_values

    def recognition_generation(self, visible_values, n_associative=20):
        """Perform one full pass of the data up and down the DBN.

        Args:
            visible_values (obj): NumPy array containing visible node values.

        Returns:
            obj: NumPy array containing new visible node activations
            obj: NumPy array containing new visible node values.

        """
        hidden_values = self.forward_pass(visible_values, n_associative / 2)
        visible_values = self.backward_pass(hidden_values, n_associative / 2)
        return visible_values

    def greedy_training(self, x, k=10, epochs=100, batch_size=100):
        """Perform greedy layer-wise training of DBN.

        Args:
            x (obj): NumPy array of dimension (n, d) containing data.
            k (int): Number of Gibbs sampling steps used during training.
            Emperically 10 is usually good.
            epochs (int): Number of training epochs.
            batch_size (int): Size of training batch.
        """
        n_l = len(self.hidden_layers_spec)
        r = RestrictedBoltzmannMachine(self.hidden_layers_spec[0],
                                       self.alpha, self.eta)
        r.contrastive_divergence(x, k, epochs, batch_size)
        _, x = r.forward_pass(x.T)
        x = x.T
        self.layers.append(r)
        for i in range(1, n_l):
            r = RestrictedBoltzmannMachine(self.hidden_layers_spec[i],
                                           self.alpha, self.eta)
            r.contrastive_divergence(x, k, epochs, batch_size, dbn_bias=False)
            _, x = r.forward_pass(x.T)
            x = x.T

            self.layers.append(r)

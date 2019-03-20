"""Class for creating and updating 2D Ising model.

"""
import matplotlib.pyplot as plt
import numpy as np


class Ising:
    """2D Ising model.

    Attributes:
        lattice (obj): (size, size) NumPy array containing spin values.
        size (int): Lattice size.
        temp (float): Temperature.

    """
    def __init__(self, temp, size):
        """Pass initial settings (T, size) to class.
        Args:
            temp (float): Temperature.
            size (int): Lattice size.
        """
        self.lattice = (2 * np.random.randint(low=0, high=2, size=(size, size))
                        - 1)
        self.size = size
        self.temp = temp

    def visualize(self, path=None):
        """Visualize and display lattice.

        Args:
            path (str): Path to save image to.

        Returns:
            type: Description of returned object.

        """
        plt.imshow(self.lattice, cmap='Greys')
        if path is not None:
            plt.savefig(path, dpi=400, bbox_inches='tight')
        plt.show()

    def get_h(self):
        """Evaluate Ising spin hamiltonian for lattice.

        Returns:
            float: Energy evaluation.

        """
        ss = self.size
        hamiltonian = 0
        for i in range(ss):
            for j in range(ss):
                if i == ss - 1:
                    right = self.lattice[i, j] * self.lattice[0, j]
                else:
                    right = self.lattice[i, j] * self.lattice[i + 1, j]
                left = self.lattice[i, j] * self.lattice[i - 1, j]
                if j == ss - 1:
                    below = self.lattice[i, j] * self.lattice[i, 0]
                else:
                    below = self.lattice[i, j] * self.lattice[i, j + 1]
                above = self.lattice[i, j] * self.lattice[i, j - 1]
                hamiltonian += -1 * float(above + below + left + right)
        hamiltonian = float(hamiltonian)
        return hamiltonian

    def get_m(self):
        """Evaluate lattice magnetization.

        Returns:
            float: Magnetization evaluation.

        """
        mag = np.sum(self.lattice)
        mag = float(mag)
        return mag

    def metropolis_update(self):
        """Perform single Metropolis-Hastings update step.

        """
        ss = self.size
        ind = np.random.randint(low=0, high=self.size, size=(2, 1))

        i = ind[0, 0]
        j = ind[1, 0]

        if i == ss - 1:
            right = self.lattice[i, j] * self.lattice[0, j]
        else:
            right = self.lattice[i, j] * self.lattice[i + 1, j]
        left = self.lattice[i, j] * self.lattice[i - 1, j]
        if j == ss - 1:
            below = self.lattice[i, j] * self.lattice[i, 0]
        else:
            below = self.lattice[i, j] * self.lattice[i, j + 1]
        above = self.lattice[i, j] * self.lattice[i, j - 1]
        delta_e = 2 * float(above + below + left + right)
        if delta_e <= 0 or np.random.random() < np.exp(-delta_e / (self.temp)):
            self.lattice[i, j] = -self.lattice[i, j]

    def run(self, iterations):
        """Run Metropolis-Hastings Monte Carlo.

        Args:
            iterations (int): Number of update steps to perform.

        """
        for _ in range(iterations):
            self.metropolis_update()


R = Ising(10, 16)
R.visualize()
R.run(10000)
R.visualize('images/progress_update_high_temp.png')

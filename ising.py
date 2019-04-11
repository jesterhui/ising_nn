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

    def wolff_update(self):
        """Perform Wolff cluster update step.

        """

        ind = np.random.randint(low=0, high=self.size, size=(2, 1))

        i = ind[0, 0]
        j = ind[1, 0]

        root = (i,j)
        self.build_cluster(root, self.lattice[root])

    def build_cluster(self, site, spin):
        """Build cluster of sites for Wolff update and flips them through recursive call.

        Args:
            site (tuple): growing point for cluster.

            spin (int): starting spin for site.

        """
        self.lattice[site] = -self.lattice[site]  # flip the spin of this site
        ss = self.size
        neighbors = [(0, 0), (0, 0), (0, 0), (0, 0)]  # define the neighbors for this site
        (i, j) = site

        if i == ss - 1:
            neighbors[0] = (0, j)
        else:
            neighbors[0] = (i + 1, j)
        if i == 0:
            neighbors[1] = (ss - 1, j)
        else:
            neighbors[1] = (i - 1, j)
        if j == ss - 1:
            neighbors[2] = (i, 0)
        else:
            neighbors[2] = (i, j + 1)
        if j == 0:
            neighbors[3] = (i, ss - 1)
        else:
            neighbors[3] = (i, j - 1)

        for next_site in neighbors:  # loop over nearest neighbors
            if self.lattice[next_site] == spin:  # excludes sites already visited which have already been flipped
                if np.random.random() < 1 - np.exp(-2.0/self.temp):
                    self.build_cluster(next_site, spin)  # recursively call build_cluster for new sites

    def run(self, iterations):
        """Run Markov Chain Monte Carlo.

        Args:
            iterations (int): Number of Monte Carlo steps to perform.
            One Monte Carlo steps includes N Metropolis updates and one Wolff
            update, where N is the number of lattice sites.

        """
        sites = (self.size)**2

        for ii in range(iterations):
            for _ in range(sites):
                self.metropolis_update()
            self.wolff_update()


R = Ising(2.7, 16)
R.visualize()
R.run(40)
R.visualize(images / wolff_crit_temp.png)

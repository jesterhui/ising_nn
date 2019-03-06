import matplotlib.pyplot as plt
import numpy as np

class Ising:
    def __init__(self, T, size):
        self.lattice = (2 * np.random.randint(low=0, high=2, size=(size, size))
                        - 1)
        self.size = size
        self.T = T


    def visualize(self):
        plt.imshow(self.lattice, cmap='Greys')
        plt.show()

    def get_h(self, lattice):
        h = 0
        for i in range(self.size):
            for j in range(self.size):
                try:
                    right = lattice[i, j] * lattice[i + 1, j]
                except IndexError:
                    right = lattice[i, j] * lattice[0, j]
                left = lattice[i, j] * lattice[i - 1, j]
                try:
                    down = lattice[i, j] * lattice[i, j + 1]
                except IndexError:
                    down = lattice[i, j] * lattice[i, 0]
                up = lattice[i, j] * lattice[i, j - 1]
                h += -1 * float(up + down + left + right)
        h = float(h)
        return h


    def get_m(self):
        m = np.sum(self.lattice)
        return m

    def update_lattice(self):
        ind = np.random.randint(low=0, high=self.size, size=(2, 1))
        new_lattice = self.lattice.copy()

        i = ind[0, 0]
        j= ind[1, 0]

        print(i, j)
        try:
            right = self.lattice[i, j] * self.lattice[i + 1, j]
        except IndexError:
            right = self.lattice[i, j] * self.lattice[0, j]
        left = self.lattice[i, j] * self.lattice[i - 1, j]
        try:
            down = self.lattice[i, j] * self.lattice[i, j + 1]
        except IndexError:
            down = self.lattice[i, j] * self.lattice[i, 0]
        up = self.lattice[i, j] * self.lattice[i, j - 1]
        delta_e = 2 * float(up + down + left + right)
        accept = None
        if delta_e <= 0:
            accept = True
        elif np.random.random() < np.exp(-delta_e / (self.T)):
            accept = True
        else:
            accept = False
        print(i, j)
        if accept is True:
            print(self.lattice[i, j])
            self.lattice[i, j] = -self.lattice[i, j]

    def run(self, iterations):
        for i in range(iterations):
            self.update_lattice()

r = Ising(2.27, 16)
r.visualize()
r.run(10000)
r.visualize()

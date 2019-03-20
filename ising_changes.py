"""Class for creating and updating 2d Ising model.

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
    def __init__(self, T, size):
        """Pass initial settings (T, size) to class.
        Args:
            temp (float): Temperature.
            size (int): Lattice size.
        """
        self.lattice = (2 * np.random.randint(low=0, high=2, size=(size, size))
                        - 1)
        self.size = size
        self.T = T


    def visualize(self):
        """Visualize and display lattice.

        """
        plt.imshow(self.lattice, cmap='Greys')
        plt.show()

    def get_h(self, lattice):
        """Evaluate Ising spin hamiltonian for lattice.

        Returns:
            float: Energy evaluation.

        """
        hamiltonian = 0
        ss = self.size
        for i in range(ss):
            for j in range(ss):
                if i == ss-1:
                    right = lattice[i, j] * lattice[0, j]
                else:
                    right = lattice[i, j] * lattice[i + 1, j] #computationally less expensive
                #try:
                #    right = lattice[i, j] * lattice[i + 1, j]
                #except IndexError:
                #    right = lattice[i, j] * lattice[0, j]
                if i == 0:
                    left = lattice[i, j] * lattice[ss-1, j]
                else:
                    left = lattice[i, j] * lattice[i - 1, j]
                    
                if j == ss-1:
                    down = lattice[i, j] * lattice[i, 0]
                else:
                    down = lattice[i, j] * lattice[i, j + 1]
                #try:
                #    down = lattice[i, j] * lattice[i, j + 1]
                #except IndexError:
                #    down = lattice[i, j] * lattice[i, 0]
                if j == 0:
                    up = lattice[i, j] * lattice[i, ss-1]
                else:
                    up = lattice[i, j] * lattice[i, j - 1]
                hamiltonian += -1 * float(up + down + left + right)
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
        ind = np.random.randint(low=0, high=ss, size=(2, 1))
        #new_lattice = self.lattice.copy() #We don't use this anywhere if I understand correctly

        i = ind[0, 0]
        j = ind[1, 0]

        print(i, j)

        if i == ss-1:
            right = self.lattice[i, j] * self.lattice[0, j]
        else:
            right = self.lattice[i, j] * self.lattice[i + 1, j]
            
            
        #try:
        #    right = self.lattice[i, j] * self.lattice[i + 1, j]
        #except IndexError:
        #    right = self.lattice[i, j] * self.lattice[0, j]
        
        
        if i == 0:
            left = self.lattice[i, j] * self.lattice[ss-1, j]
        else: 
            left = self.lattice[i, j] * self.lattice[i - 1, j]
            
            
        if j == ss-1:
            down = self.lattice[i, j] * self.lattice[i, 0]
        else:
            down = self.lattice[i, j] * self.lattice[i, j + 1]
            
            
        #try:
        #    down = self.lattice[i, j] * self.lattice[i, j + 1]
        #except IndexError:
        #    down = self.lattice[i, j] * self.lattice[i, 0]
        
        
        if j == 0:
            up = self.lattice[i, j] * self.lattice[i, ss-1]
        else:
            up = self.lattice[i, j] * self.lattice[i, j - 1]
            
            
        delta_e = 2 * float(up + down + left + right)
        #accept = None
        print(i, j)
        if (delta_e <= 0 or np.random.random() < np.exp(-delta_e / (self.T))): #Merged the if-else statements together.
            print(self.lattice[i, j])
            self.lattice[i, j] = -self.lattice[i, j]

    def run(self, iterations):
        """Run Metropolis-Hastings Monte Carlo.

        Args:
            iterations (int): Number of update steps to perform.

        """
        for i in range(iterations):
            self.metropolis_update()

r = Ising(2.27, 16)
r.visualize()
r.run(10000)
r.visualize()

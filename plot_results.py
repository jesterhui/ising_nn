import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def get_energies(array):
    energies = np.zeros((10000, 1))
    for ind, lattice in enumerate(array):
        lattice = 2 * lattice - 1
        lattice = lattice.reshape(8, 8)
        hamiltonian = 0
        for i in range(8):
            for j in range(8):
                if i == 7:
                    right = lattice[i, j] * lattice[0, j]
                else:
                    right = lattice[i, j] * lattice[i + 1, j]
                left = lattice[i, j] * lattice[i - 1, j]
                if j == 7:
                    below = lattice[i, j] * lattice[i, 0]
                else:
                    below = lattice[i, j] * lattice[i, j + 1]
                above = lattice[i, j] * lattice[i, j - 1]
                hamiltonian += -1 * float(above + below + left + right)
        energies[ind] = float(hamiltonian)
    return energies


mpl.rc('font', **{'family': 'serif'})
mpl.rc('text', usetex=True)

RBN_ENERGY = np.zeros((10000, 1))
DBN_ENERGY = np.zeros((10000, 1))
MONTE_CARLO_AVERAGE = []
MONTE_CARLO_C = []
RBN_AVERAGE = []
RBN_C = []
DBN_AVERAGE = []
DBN_C = []
for temp in np.linspace(1.0, 3.5, 26):
    TRAINING_DATA = np.load('data/train_temp_%g.npy' % (temp))
    RBN_DATA = np.load('data/test_rbn_temp_%g.npy' % (temp))
    DBN_DATA = np.load('data/test_dbn_temp_%g.npy' % (temp))
    MONTE_CARLO = get_energies(TRAINING_DATA)
    MONTE_CARLO_AVERAGE.append(np.mean(MONTE_CARLO))
    MONTE_CARLO_C.append((np.mean(MONTE_CARLO ** 2)
                          - np.mean(MONTE_CARLO) ** 2) / (temp ** 2))
    RBN = get_energies(RBN_DATA)
    RBN_AVERAGE.append(np.mean(RBN))
    RBN_C.append((np.mean(RBN ** 2) - np.mean(RBN) ** 2) / (temp ** 2))
    DBN = get_energies(DBN_DATA)
    DBN_AVERAGE.append(np.mean(DBN))
    DBN_C.append((np.mean(DBN ** 2) - np.mean(DBN) ** 2) / (temp ** 2))
    if temp == 2:
        FIG, AX = plt.subplots(1, 3, figsize=(6.5, 2))
        AX[0].hist(MONTE_CARLO, color='k', label='MC')
        AX[0].set_title('MC')
        AX[0].set_xlabel('Energy')
        AX[1].hist(RBN, color='#e41a1c',
                   label='Restricted Boltzmann Machine')
        AX[1].set_title('RBM')
        AX[1].set_xlabel('Energy')
        AX[2].hist(DBN, color='#377eb8', label='Deep Belief Network')
        AX[2].set_title('DBN')
        AX[2].set_xlabel('Energy')
        for i in range(3):
            plt.setp(AX[i].spines.values(), linewidth=3, zorder=30)
            AX[i].tick_params(axis='both', width=2, direction='in', pad=5)
            AX[i].set_xlim(-275, 0)
            AX[i].set_ylim(0, 3500)
        plt.tight_layout()
        plt.savefig('images/histograms.png', dpi=400,
                    bbox_inches='tight')
        plt.show()
FIG, AX = plt.subplots()
plt.plot(np.linspace(1.0, 3.5, 26), MONTE_CARLO_AVERAGE, linewidth=1.5,
         color='k', label='Monte Carlo')
plt.scatter(np.linspace(1.0, 3.5, 26), RBN_AVERAGE, c='#e41a1c',
            label='Restricted Boltzmann Machine', edgecolors='k')
plt.scatter(np.linspace(1.0, 3.5, 26), DBN_AVERAGE, c='#377eb8',
            label='Deep Belief Network', marker='s', edgecolors='k')
plt.setp(AX.spines.values(), linewidth=3, zorder=30)
AX.tick_params(axis='both', width=2, direction='in', pad=5)
plt.xlabel('Temperature')
plt.ylabel('Average energy per spin')
AX.legend(frameon=False)
plt.savefig('images/energy_plot.png', dpi=400, bbox_inches='tight')
plt.show()
FIG, AX = plt.subplots()
plt.plot(np.linspace(1.0, 3.5, 26), MONTE_CARLO_C, linewidth=1.5,
         color='k', label='Monte Carlo')
plt.scatter(np.linspace(1.0, 3.5, 26), RBN_C, c='#e41a1c',
            label='Restricted Boltzmann Machine', edgecolors='k')
plt.scatter(np.linspace(1.0, 3.5, 26), DBN_C, c='#377eb8',
            label='Deep Belief Network', marker='s', edgecolors='k')
plt.setp(AX.spines.values(), linewidth=3, zorder=30)
AX.tick_params(axis='both', width=2, direction='in', pad=5)
plt.xlabel('Temperature')
plt.ylabel('Average heat capacity per spin')
AX.legend(frameon=False)
plt.savefig('images/heat_capacity_plot.png', dpi=400, bbox_inches='tight')
plt.show()

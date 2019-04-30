import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rc('font', **{'family': 'serif'})
mpl.rc('text', usetex=True)
FIG, AX = plt.subplots(3, 3)
for i, temp in enumerate([1, 2.1, 3.5]):
    TRAINING_DATA = np.load('data/train_temp_%g.npy' % (temp))
    RBN_DATA = np.load('data/test_rbn_temp_%g.npy' % (temp))
    DBN_DATA = np.load('data/test_dbn_temp_%g.npy' % (temp))
    for j, DATA in enumerate([TRAINING_DATA, RBN_DATA, DBN_DATA]):
        ind = np.random.randint(10000)
        AX[i, j].imshow(DATA[ind].reshape(8, 8), cmap='Greys')
        AX[i, j].set_xticks([])
        AX[i, j].set_yticks([])
        plt.setp(AX[i, j].spines.values(), linewidth=3, zorder=30)
AX[2, 0].set_xlabel('MC')
AX[2, 1].set_xlabel('RBM')
AX[2, 2].set_xlabel('DBN')
AX[0, 0].set_ylabel('$T=1.0$')
AX[1, 0].set_ylabel('$T=2.1$')
AX[2, 0].set_ylabel('$T=3.5$')
plt.tight_layout()
plt.savefig('method_comparison.png', dpi=400, bbox_inches='tight')
plt.show()

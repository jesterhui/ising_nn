import numpy as np
from boltzmann_machines import RestrictedBoltzmannMachine

DATA_POINTS = 10000
SIZE = 8

for temp in np.linspace(1.0, 3.5, 26):
    GENERATED_DATA = np.zeros(shape=(DATA_POINTS, SIZE ** 2))
    TRAINING_DATA = np.load('data/train_temp_%g.npy' % (temp))
    R = RestrictedBoltzmannMachine(32)
    R.contrastive_divergence(TRAINING_DATA)
    _, SAMPLE = R.gibbs(TRAINING_DATA[0, :].reshape(64, 1))
    for i in range(DATA_POINTS):
        _, SAMPLE = R.gibbs(SAMPLE)
        GENERATED_DATA[i] = SAMPLE[1:].reshape(1, SIZE ** 2)
    np.save('data/test_rbn_temp_%g' % (temp), GENERATED_DATA)
    print('T = {} data collected'.format(temp))

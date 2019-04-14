import numpy as np
from boltzmann_machines import DeepBeliefNetwork

DATA_POINTS = 10000
SIZE = 8

for temp in np.linspace(1.0, 3.5, 26):
    GENERATED_DATA = np.zeros(shape=(DATA_POINTS, SIZE ** 2))
    TRAINING_DATA = np.load('data/train_temp_%g.npy' % (temp))
    R = DeepBeliefNetwork([24, 24])
    R.greedy_training(TRAINING_DATA)
    for i in range(DATA_POINTS):
        RANDOM = np.random.randint(1, size=(24, 1))
        SAMPLE = R.backward_pass(RANDOM)
        GENERATED_DATA[i] = SAMPLE[1:].reshape(1, SIZE ** 2)
    np.save('data/test_dbn_temp_%g' % (temp), GENERATED_DATA)
    print('T = {} data collected'.format(temp))

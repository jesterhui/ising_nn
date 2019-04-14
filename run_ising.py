import numpy as np
from ising import Ising

DATA_POINTS = 10000
SIZE = 8

for temp in np.linspace(1.0, 3.5, 26):
    data = np.zeros(shape=(DATA_POINTS, SIZE ** 2))
    R = Ising(temp, SIZE)
    R.run(32 ** 3)
    print('T = {} equilibration done'.format(temp))
    for i in range(DATA_POINTS):
        R = Ising(temp, SIZE)
        R.run(32)
        data[i] = R.generate_data()
    np.save('data/train_temp_%g' % (temp), data)
    print('T = {} data collected'.format(temp))

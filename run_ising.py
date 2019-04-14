import numpy as np
from ising import Ising

datapoints = 10000
size = 8

for temp in np.linspace(1.0, 3.5, 26):
    data = np.zeros(shape=(datapoints, size ** 2))
    R = Ising(temp, size)
    R.run(32 ** 3)
    print('T = {} equilibration done'.format(temp))
    for i in range(datapoints):
        R = Ising(temp, size)
        R.run(32)
        data[i] = R.generate_data()
    np.save('data/train_temp_%g' % (temp), data)
    print('T = {} data collected'.format(temp))

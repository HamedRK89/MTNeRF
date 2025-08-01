import numpy as np

data = np.load('poses_bounds_9_9.npy')

data_orig = data[0:9]
data_virtual = data[9:18]

print(data_orig.shape)
print(data_virtual.shape)

np.save('poses_bounds_orig.npy', data_orig)
np.save('poses_bounds_virtual.npy', data_virtual)
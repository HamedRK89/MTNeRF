import numpy as np
'''
data = np.load('poses_bounds_orig.npy')
print(data)
'''

data = np.load('poses_bounds.npy')
#print(data)
data_orig_num = data[[0, 1, 4, 12, 20, 24]]
data_virtual_num = data[[2, 6, 8, 10, 14, 16, 17, 18, 22]]
data_mix = data[[0, 1, 4, 12, 20, 24, 2, 6, 8, 10, 14, 16, 17, 18, 22]]
np.save('poses_bounds_orig.npy', data_orig_num)
np.save('poses_bounds_virtual.npy', data_virtual_num)
np.save('poses_bounds_mix.npy', data_mix)
#print(data_mix)

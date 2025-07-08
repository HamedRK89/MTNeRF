import imageio.v3 as iio
import imageio

depth_iio = iio.imread('00.png', mode='F')
#print(depth_iio)
print(depth_iio.dtype)
print(depth_iio.shape)

depth = imageio.imread('00.png', ignoregamma=True)
print(depth)
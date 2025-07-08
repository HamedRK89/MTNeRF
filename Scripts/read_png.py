from PIL import Image
import numpy as np

img = Image.open("00_RGBA.png")
img_np = np.array(img)
print(img_np.shape)
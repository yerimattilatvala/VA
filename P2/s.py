import numpy as np
from skimage.draw import rectangle
img = np.zeros((5, 5), dtype=np.uint8)
start = (1, 1)
extent = (3, 3)
rr, cc = rectangle(start, extent=extent, shape=img.shape)
img[rr, cc] = 1
img
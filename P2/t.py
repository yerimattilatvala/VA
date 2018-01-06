import cv2
import numpy as np


def is_b_or_w(image, black_max_bgr=(40, 40, 40)):
    # use this if you want to check channels are all basically equal
    # I split this up into small steps to find out where your error is coming from
    mean_bgr_float = np.mean(image, axis=(0,1))
    mean_bgr_rounded = np.round(mean_bgr_float)
    mean_bgr = mean_bgr_rounded.astype(np.uint8)
    # use this if you just want a simple threshold for simple grayscale
    # or if you want to use an HSV (V) measurement as in your example
    mean_intensity = int(round(np.mean(image)))
    return 'black' if np.all(mean_bgr < black_max_bgr) else 'white'

# make a test image for ROIs
shape = (10, 10, 3)  # 10x10 BGR image
im_blackleft_white_right = np.ndarray(shape, dtype=np.uint8)
im_blackleft_white_right[:, 0:4] = 10
im_blackleft_white_right[:, 5:9] = 255

roi_darkgray = im_blackleft_white_right[:,0:4]
roi_white = im_blackleft_white_right[:,5:9]
img = cv2.imread('4.Eyes_and_gaze\Derecha1.jpg')

# test them with ROI
is_b_or_w(img,roi_darkgray)
is_b_or_w(img,roi_white)

# output
# dark gray image identified as: black
# white image identified as: white
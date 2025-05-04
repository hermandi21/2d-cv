import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time

def rgb_2_gray(img, mode='lut'):
    if mode == 'lut':
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.0722)
    else:
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)
    

def sobel(img, filter):
    # TODO: implement sobel filtering e.g. with 4 foor loops
    
    filtered_img = np.zeros((gray.shape[0]-2, gray.shape[1]-2))
    return filtered_img


img = io.imread("lena.jpg")
gray = rgb_2_gray(img)

height, width = gray.shape

# TODO: define filter in x in y direction

start = time.time()
# TODO: filter image in x direction (sobel(gray, filter_x))
end = time.time()
duration = end-start
print("Duration in milliseconds: ", duration*1000)

start = time.time()
# TODO: filter image in y direction (sobel(gray, filter_y))
end = time.time()
duration = end-start
print("Duration in milliseconds: ", duration*1000)


# TODO compute Gradient magnitude

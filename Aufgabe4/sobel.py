import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time

def rgb_2_gray(img, mode='lut'):
    if mode == 'lut':
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.0722)
    else:
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)
    

def sobel(img, filter_kernel):
    height, width = img.shape  # Bilddimensionen
    kernel_height, kernel_width = filter_kernel.shape  # Filterdimensionen

    output_height = height - kernel_height + 1  # Höhe des gefilterten Bildes
    output_width = width - kernel_width + 1  # Breite des gefilterten Bildes

    output = np.zeros((output_height, output_width))  # Ausgabe-Array initialisieren

    # Filterung
    for i in range(output.shape[0]):  # Höhe des Ausgabe-Arrays
        for j in range(output.shape[1]):  # Breite des Ausgabe-Arrays
            # initialisiere Summe
            convolution_sum = 0.0

            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    pixel_wert = img[i + ki, j + kj]
                    kernel_value = filter_kernel[ki, kj]
                    convolution_sum += pixel_wert * kernel_value

            output[i, j] = convolution_sum

    return output
    

# Sobel-Filter für die Kantendetektion

# Filter für die x-Richtung (detektiert vertikale Kanten)
filter_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Filter für die y-Richtung (detektiert horizontale Kanten)
filter_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])



# Kantenstaerke berechenne
def calc_gradient_magnitude(Ix, Iy):
    # Berechne die Magnitude der Gradienten
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)

    gradient_magnitude /= np.max(gradient_magnitude)
    return gradient_magnitude



import sobel_demo as nd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time


def rgb_2_gray(img, mode='lut'):
    if mode == 'lut':
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.0722)
    else:
        return np.round(img[:,:,0] * 0.2126 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)

def gradient_magnitude(dx, dy):
    """
    Berechnet elementweise den Betrag des Gradienten ohne externe Bibliotheken.
    
    Args:
        dx: 2D-Liste (Liste von Listen) mit den x-Ableitungswerten.
        dy: 2D-Liste (Liste von Listen) mit den y-Ableitungswerten.
             dx und dy müssen die gleiche Größe haben.
    
    Returns:
        2D-Liste gleicher Dimension wie dx/dy, 
        wobei jeder Eintrag sqrt(dx[i][j]**2 + dy[i][j]**2) ist.
    """
    # Höhe und Breite ermitteln
    h = len(dx)
    if h == 0:
        return []
    w = len(dx[0])
    
    # Ausgabe-Array vorbereiten
    mag = [[0] * w for _ in range(h)]
    
    # Elementweise Berechnung
    for i in range(h):
        for j in range(w):
            # Quadrieren, aufsummieren und Wurzel ziehen
            mag[i][j] = (dx[i][j] * dx[i][j] + dy[i][j] * dy[i][j]) ** 0.5
    
    return mag


img = io.imread("lena.jpg")
gray = rgb_2_gray(img).astype("float64")

# ----------TODO: define filter in x in y direction-------------
# Sobel-Kern für Ableitung in x-Richtung
filter_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

# Sobel-Kern für Ableitung in y-Richtung
filter_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=float)


# -----------TODO: filter image in x direction (sobel(gray, filter_x))-----------
start = time.time()
dx_cpp = nd.sobel(gray, filter_x)
end = time.time()
duration = end-start
print("Duration in milliseconds for x direction: ", duration*1000)

# -----------TODO: filter image in y direction (sobel(gray, filter_y))-----------
start = time.time()
dy_cpp = nd.sobel(gray, filter_y)
end = time.time()
duration = end-start
print("Duration in milliseconds for y direction: ", duration*1000)


# -----------TODO compute Gradient magnitude-----------
mag = gradient_magnitude(dx_cpp, dy_cpp)
mag8 = (mag/mag.max()*255).astype(np.uint8)
print("Gradient magnitude:", mag.shape)


#-----------TODO: plot image-----------
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.imshow(mag8, cmap="gray")
ax.set_title("Sobel C++ Gradient")
ax.axis("off")
plt.show()
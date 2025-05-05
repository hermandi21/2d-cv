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
    h, w = img.shape
    # Ausgabe ist um je 1 Pixel an jeder Seite kleiner
    out = np.zeros((h-2, w-2), dtype=float)
    for i in range(1, h-1):
        for j in range(1, w-1):
            sum = 0.0
            for u in range(-1, 2):
                for v in range(-1, 2):
                    sum += filter[u+1, v+1] * img[i+u, j+v]
            out[i-1, j-1] = sum
    return out

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
gray = rgb_2_gray(img)

height, width = gray.shape

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
dx = sobel(gray, filter_x)
end = time.time()
duration = end-start
print("Duration in milliseconds for x direction: ", duration*1000)

# -----------TODO: filter image in y direction (sobel(gray, filter_y))-----------
start = time.time()
dy = sobel(gray, filter_y)
end = time.time()
duration = end-start
print("Duration in milliseconds for y direction: ", duration*1000)



# --- Kantenstaerke via Funktion berechnen und ausgeben ---
edge_strength = gradient_magnitude(dx, dy)
print("Gradient magnitude:", edge_strength)


# --- Normierung auf 0–255 und Umwandlung in uint8 ---


# --- Visualisierung ---
fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(np.abs(dx), cmap="gray")
axes[0].set_title("Sobel X"); axes[0].axis("off")
axes[1].imshow(np.abs(dy), cmap="gray")
axes[1].set_title("Sobel Y"); axes[1].axis("off")
axes[2].imshow(edge_strength, cmap="gray")
axes[2].set_title("Kantenstaerke"); axes[2].axis("off")
plt.tight_layout()
plt.show()

# --- Ergebnisse speichern ---
#io.imsave("sobel_x.png", (np.abs(dx)/dx.max()*255).astype(np.uint8))
#io.imsave("sobel_y.png", (np.abs(dy)/dy.max()*255).astype(np.uint8))
#io.imsave("sobel_strength.png", edge_strength)

import numpy as np
from skimage.util import view_as_windows


def filter1(img, filter, off):
    """
    Faltet ein Bild mit einer gegebenen Filtermaske und einem Offset.

        off (int): Offset (Stride), gibt an, um wie viele Pixel die Filtermatrix verschoben wird.

    """
    # Filtergröße
    filter_size = filter.shape[0]
    assert filter_size % 2 == 1, "Die Filtermatrix muss eine ungerade Größe haben (NxN mit N = 2K + 1)."
    assert len(img.shape) == 2, "Das Eingabebild muss ein Graustufenbild sein."

    # Bild in Fenster zerlegen
    windows = view_as_windows(img, (filter_size, filter_size), step=off)

    # Faltung durchführen
    result = np.tensordot(windows, filter, axes=((2, 3), (0, 1)))

    # Ergebnis auf gültigen Bereich (0-255) beschränken und in int umwandeln
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def filter2(in_image, filter_matrix, offset, edge='min'):
    """
    Faltet ein Bild mit einer gegebenen Filtermatrix, Offset (Stride) und Randbehandlung.

    Args:
        in_image (numpy.ndarray): 8-bit Graustufenbild.
        filter_matrix (numpy.ndarray): Filtermatrix (float) der Größe NxN (N = 2K+1).
        offset (int): Offset (Stride).
        edge (str, optional): Randbehandlungsmethode ('min', 'max', 'continue'). 
                             Defaults to 'min' (Zero Padding).

    Returns:
        numpy.ndarray: Gefaltetes Bild (8-bit Graustufenbild).
    """

    # Sicherstellen, dass die Filtermatrix eine ungerade Größe hat
    filter_size = filter_matrix.shape[0]
    assert filter_size == filter_matrix.shape[1], "Filtermatrix muss quadratisch sein."
    assert filter_size % 2 != 0, "Filtermatrix muss eine ungerade Anzahl an Zeilen/Spalten haben."

    # Padding berechnen und Randbehandlung anwenden
    pad_size = filter_size // 2
    if edge == 'min':
        padded_image = np.pad(in_image, pad_size, mode='constant', constant_values=0)
    elif edge == 'max':
        padded_image = np.pad(in_image, pad_size, mode='constant', constant_values=255)
    elif edge == 'continue':
        padded_image = np.pad(in_image, pad_size, mode='edge')  # 'edge' entspricht 'continue'
    else:
        raise ValueError("Ungültige Randbehandlungsmethode. Wählen Sie zwischen 'min', 'max' oder 'continue'.")

    # Bild in Fenster zerlegen
    windows = view_as_windows(padded_image, (filter_size, filter_size), step=offset)

    # Faltung durchführen
    out_image = np.tensordot(windows, filter_matrix, axes=((-2, -1), (0, 1)))

    # Sicherstellen, dass die Werte im gültigen Bereich liegen (0-255)
    out_image = np.clip(out_image, 0, 255).astype(np.uint8)

    return out_image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
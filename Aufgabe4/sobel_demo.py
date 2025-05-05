import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time
import sobel_demo as nd # Import the C++ module
from sobel import rgb_2_gray # Import the missing function

# Load the image
img = io.imread("lena.jpg")

# Convert image to grayscale
gray = rgb_2_gray(img).astype("float64")

# Define Sobel filters
filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float64")
filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="float64")


start = time.time()
# Filter image in x direction
filtered_x = nd.sobel(gray, filter_x)
end = time.time()
duration = end-start
print("Duration C++ (x-direction) in milliseconds: ", duration*1000)


start = time.time()
# Filter image in y direction
filtered_y = nd.sobel(gray, filter_y)
end = time.time()
duration = end-start
print("Duration C++ (y-direction) in milliseconds: ", duration*1000)

# Compute Gradient magnitude
gradient_magnitude = np.sqrt(filtered_x**2 + filtered_y**2)

# Normalize magnitude for display
gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray[1:-1, 1:-1], cmap='gray') # Adjust gray image size to match filtered output
plt.title('Original (Cropped)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Sobel Gradient Magnitude (C++)')
plt.axis('off')

# Optional: Display individual x and y gradients
# Normalize gradients for display
filtered_x_display = ((filtered_x - np.min(filtered_x)) / (np.max(filtered_x) - np.min(filtered_x)) * 255).astype(np.uint8)
filtered_y_display = ((filtered_y - np.min(filtered_y)) / (np.max(filtered_y) - np.min(filtered_y)) * 255).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(filtered_x_display, cmap='gray')
plt.title('Sobel X Gradient (C++)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_y_display, cmap='gray')
plt.title('Sobel Y Gradient (C++)')
plt.axis('off')


plt.show()

"""
py -m pip install opencv-python numpy matplotlib
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read image
image = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE)


# 3x3 mask
mask = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Get image dimensions
rows, cols = image.shape

# Create output image
output = np.zeros((rows, cols), dtype=np.float32)

# Perform convolution manually
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        region = image[i-1:i+2, j-1:j+2]
        output[i, j] = np.sum(region * mask)

# Normalize result for display
output = cv2.normalize(
    output, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Display results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray")
plt.title("After 3x3 Convolution")
plt.axis("off")

plt.tight_layout()
plt.show()

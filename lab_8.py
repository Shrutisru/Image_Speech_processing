"""
py -m pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read original image in color
img_color = cv2.imread("cat.png")

# Convert a copy to grayscale for image processing
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# i) Threshold Image
_, threshold = cv2.threshold(
    img_gray, 127, 255, cv2.THRESH_BINARY
)

# ii) Power-law Enhancement
gamma = 0.5

normalized = img_gray / 255.0
power_img = np.power(normalized, gamma)
power_img = np.uint8(power_img * 255)

# iii) High-pass Image
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

high_pass = cv2.filter2D(img_gray, -1, kernel)

# Display Results
plt.figure(figsize=(12, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Threshold
plt.subplot(2, 2, 2)
plt.imshow(threshold, cmap="gray")
plt.title("Threshold Image")
plt.axis("off")

# Power-law
plt.subplot(2, 2, 3)
plt.imshow(power_img, cmap="gray")
plt.title("Power-law Enhanced Image")
plt.axis("off")

# High-pass
plt.subplot(2, 2, 4)
plt.imshow(high_pass, cmap="gray")
plt.title("High-pass Image")
plt.axis("off")

plt.tight_layout()
plt.show()
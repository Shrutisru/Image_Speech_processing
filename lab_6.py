"""
py -m pip install opencv-python numpy matplotlib
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE)

# Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

# Laplacian Edge Detection
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# Canny Edge Detection
canny = cv2.Canny(image, 100, 200)

# Display and Compare Results
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Edge Detection")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()
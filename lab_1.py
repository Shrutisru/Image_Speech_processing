"""
py -m pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cat.png')

# Convert BGR to RGB and grayscale
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# i) Histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# ii) Low Pass Filter
low_pass = cv2.GaussianBlur(gray, (9, 9), 0)

# iii) High Pass Filter
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
high_pass = cv2.filter2D(gray, -1, kernel)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.imshow(low_pass, cmap='gray')
plt.title('Low Pass Filtered Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(high_pass, cmap='gray')
plt.title('High Pass Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()

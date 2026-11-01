"""
Install: py -m pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image
image = cv2.imread("character.png", cv2.IMREAD_GRAYSCALE)

# 2. Convert to binary image
_, binary = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 3. Remove small gaps using Morphological Closing
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 4. Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, 8
)

# 5. Segment individual characters
characters = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]

    if area > 500:
        char = 255 - binary[y:y+h, x:x+w]
        char = cv2.copyMakeBorder(char, 8, 8, 8, 8,
                                  cv2.BORDER_CONSTANT, value=255)
        characters.append((x, char))

# Sort characters from left to right
characters.sort(key=lambda item: item[0])

print("Number of characters:", len(characters))

# 6. Display results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(255 - binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# 7. Display segmented characters
plt.figure(figsize=(12, 3))

for i, (x, char) in enumerate(characters):
    plt.subplot(1, len(characters), i + 1)
    plt.imshow(char, cmap="gray")
    plt.title(f"Character {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
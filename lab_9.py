
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("cat.png")

if img is None:
    raise FileNotFoundError("cat.png not found")

# Original image
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Histogram Equalization
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

# Smoothing Filters
mean = cv2.cvtColor(cv2.blur(img, (3, 3)), cv2.COLOR_BGR2RGB)
gaussian = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2RGB)
median = cv2.cvtColor(cv2.medianBlur(img, 3), cv2.COLOR_BGR2RGB)

# Sharpening Filter
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

sharp = cv2.cvtColor(cv2.filter2D(img, -1, kernel), cv2.COLOR_BGR2RGB)

# Display Results
titles = ["Original Image", "Enhanced Image", "Mean Filter",
          "Gaussian Filter", "Median Filter", "Sharpened Image"]

images = [original, enhanced, mean, gaussian, median, sharp]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

for ax, title, image in zip(axes.flat, titles, images):
    ax.imshow(image)
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis("off")

plt.show()
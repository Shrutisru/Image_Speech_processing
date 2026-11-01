"""
py -m pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("coins.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise
gray = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=65,
    param1=100,
    param2=45,
    minRadius=40,
    maxRadius=90
)

result = img.copy()

if circles is not None:

    circles = np.round(circles[0]).astype(int)

    # Sort circles from top to bottom
    circles = sorted(circles, key=lambda c: (c[1], c[0]))

    print("Area of each coin:")

    for i, (x, y, r) in enumerate(circles, 1):

        # Area of circle
        area = np.pi * r * r

        # Draw circle
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)

        # Draw center
        cv2.circle(result, (x, y), 3, (255, 0, 0), -1)

        # Display area
        text = f"Coin {i}: {area:.0f} px"

        cv2.putText(
            result,
            text,
            (x - 60, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        print(f"Coin {i}: {area:.2f} pixels")

else:
    print("No coins detected.")

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result_rgb)
plt.title("Coins with Area")
plt.axis("off")

plt.tight_layout()
plt.show()

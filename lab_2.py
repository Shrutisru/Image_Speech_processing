"""
py -m pip install numpy matplotlib scikit-image
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, filters, measure

# 1. Read rice image
img = io.imread("rice.tif", as_gray=True)

# 2. Remove background
bg = morphology.opening(img, morphology.disk(15))
corrected = img - bg

# 3. Convert to binary image
th = filters.threshold_otsu(corrected)
bw = corrected > th

# 4. Label rice grains and remove small noise
labels = measure.label(bw)
props = [p for p in measure.regionprops(labels) if p.area >= 50]

# 5. Count rice grains
print("Total rice grains:", len(props))
print("No.   Area   Major Axis   Perimeter")

for i, p in enumerate(props, 1):
    print(i,  int(p.area), 
           round(p.axis_major_length, 2),
           round(p.perimeter, 2))

# 6. Count rice grains in a specific area range
low, high = 150, 300
selected = [p for p in props if low <= p.area <= high]

print("Rice in area range:", len(selected))

# 7. Display selected rice grains
result = np.isin(labels, [p.label for p in selected])
plt.imshow(result, cmap="gray")
plt.title("Selected Rice Grains")
plt.axis("off")
plt.show()

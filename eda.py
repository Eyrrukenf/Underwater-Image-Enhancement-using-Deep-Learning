
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

INPUT_DIR = "data/raw/UIEB/input"
GT_DIR = "data/raw/UIEB/gt"

input_images = sorted(os.listdir(INPUT_DIR))
gt_images = sorted(os.listdir(GT_DIR))

print("Total input images:", len(input_images))
print("Total ground truth images:", len(gt_images))

assert input_images == gt_images, "Input and GT images are not aligned!"
print("All image pairs are correctly aligned.")

heights = []
widths = []

for img_name in input_images:
    img = cv2.imread(os.path.join(INPUT_DIR, img_name))
    h, w, _ = img.shape
    heights.append(h)
    widths.append(w)

print("Min resolution:", min(widths), "x", min(heights))
print("Max resolution:", max(widths), "x", max(heights))

plt.figure()
plt.hist(heights, bins=20)
plt.title("Image Height Distribution")
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(widths, bins=20)
plt.title("Image Width Distribution")
plt.xlabel("Width")
plt.ylabel("Frequency")
plt.show()

sample_img = cv2.imread(os.path.join(INPUT_DIR, input_images[0]))
gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.hist(gray.flatten(), bins=256)
plt.title("Pixel Intensity Distribution (Grayscale)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

plt.figure()
plt.hist(r.flatten(), bins=256, alpha=0.5, label="Red")
plt.hist(g.flatten(), bins=256, alpha=0.5, label="Green")
plt.hist(b.flatten(), bins=256, alpha=0.5, label="Blue")
plt.legend()
plt.title("RGB Channel Distribution")
plt.show()

idx = np.random.randint(len(input_images))

input_img = cv2.cvtColor(
    cv2.imread(os.path.join(INPUT_DIR, input_images[idx])), cv2.COLOR_BGR2RGB
)
gt_img = cv2.cvtColor(
    cv2.imread(os.path.join(GT_DIR, gt_images[idx])), cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(input_img)
plt.title("Underwater Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gt_img)
plt.title("Ground Truth Image")
plt.axis("off")

plt.show()

print("Mean pixel value (input):", np.mean(sample_img))
print("Std pixel value (input):", np.std(sample_img))









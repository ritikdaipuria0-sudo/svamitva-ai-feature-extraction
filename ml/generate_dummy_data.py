import os
import cv2
import numpy as np

os.makedirs("../data/images", exist_ok=True)
os.makedirs("../data/masks", exist_ok=True)

for i in range(20):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)

    # building
    cv2.rectangle(image, (20, 20), (60, 60), (180, 180, 180), -1)
    cv2.rectangle(mask, (20, 20), (60, 60), 1, -1)

    # road
    cv2.line(image, (0, 100), (128, 80), (120, 120, 120), 8)
    cv2.line(mask, (0, 100), (128, 80), 2, 8)

    # water body
    cv2.circle(image, (95, 35), 18, (255, 0, 0), -1)
    cv2.circle(mask, (95, 35), 18, 3, -1)

    image_path = f"../data/images/image_{i}.png"
    mask_path = f"../data/masks/mask_{i}.png"

    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)

print("Dummy dataset created successfully!")
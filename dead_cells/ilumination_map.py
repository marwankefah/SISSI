#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

#%%
data_dir = Path(
    "/Users/manasikattel/Documents/cell-segmentation/data/raw/named_images_type/dead"
)
dead_images_raw = [
    cv2.imread(str(data_dir / img), cv2.IMREAD_GRAYSCALE) for img in data_dir.iterdir()
]

image = cv2.imread(str(data_dir / Path("cell57.jpg")), cv2.IMREAD_GRAYSCALE)
# %%
plt.imshow(image, cmap="gray")
# %%
_, bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.imshow(bin, cmap="gray")

#%%
IF = cv2.morphologyEx(
    image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (300, 300))
)
plt.imshow(IF, cmap="gray")

#%%

#%%
th = image - IF
plt.imshow(th, cmap="gray")

#%%
# cv::threshold(th, th, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
# aia::imshow("Binarized after tophat", th, true, 2.0);


#%%
plt.imshow(image, cmap="gray")


#%%
# Alternate filtering
img = image.copy()
for i in range(3, 50, 2):

    fig, ax = plt.subplots(1, 3, figsize=(5, 10))
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, SE)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, SE)

    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(img, cmap="gray")
    ax[2].imshow(abs(image - img), cmap="gray")
    plt.show()
# %%
cv2.imwrite("illumination_corrected.png", image + (255 - img - 60) * 0.9)

# %%
image - img
# %%
cv2.imwrite("difference.png", image - (image - img * 0.5))
# %%

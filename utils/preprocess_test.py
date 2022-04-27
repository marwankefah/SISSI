import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import morphology
from pathlib import Path

from utils.preprocess import illumination_correction, EGT_Segmentation

data_dir = Path(
    "../data/chrisi/inhib"
)
dead_images_raw = [
    cv2.imread(str(data_dir / img), cv2.IMREAD_GRAYSCALE) for img in data_dir.iterdir()
]

cell_gray = cv2.imread(str(data_dir / Path("cell4.jpg")), cv2.IMREAD_GRAYSCALE)

plt.imshow(cell_gray, cmap="gray")
plt.show()

cell_illumination_corrected = illumination_correction(cell_gray)
plt.imshow(cell_illumination_corrected, cmap="gray")
plt.show()

clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(16, 16))
cell_clahe = clahe.apply(cell_illumination_corrected.astype('uint8'))

plt.imshow(cell_clahe, cmap="gray")
plt.show()

S = EGT_Segmentation(cell_clahe, 8, 10, 2)
plt.imshow(S, cmap="gray")
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import feature
from skimage.color import rgb2gray
from skimage import measure
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.filters import gaussian
# %%
# Generate an initial image with two overlapping circles
# x, y = np.indices((80, 80))
# x1, y1, x2, y2 = 28, 28, 44, 52
# r1, r2 = 16, 20
# mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
# mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
# image = np.logical_or(mask_circle1, mask_circle2)
# %%

cell_type = 'inhib'
data_dir = Path(
    "../data/chrisi/" + cell_type + "/"
)
dead_images_raw = [
    [cv2.imread(str(img)), str(img).split('\\')[-1]] for img in data_dir.iterdir()
]
dead_images_raw.remove([None, '.gitignore'])

for image, cell_name in dead_images_raw:
    image_gray = rgb2gray(image)
    edges1 = feature.canny(image_gray, sigma=0.1)
    #edges2 = feature.canny(image_gray, sigma=3)

    #SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges1=edges1.astype(int)
    # edges1 = cv2.morphologyEx(edges1*255, cv2.MORPH_OPEN, SE)

    # label_im = label(edges1)
    # regions = regionprops(label_im)
    num_labels, labels = cv2.connectedComponents(edges1)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    # Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(image_gray)
    ax1.imshow(edges1)
    ax2.imshow(label_im)
    plt.show()


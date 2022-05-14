
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
    #edges1=edges1.astype(int)
    #edges1 = cv2.morphologyEx(edges1*255, cv2.MORPH_OPEN, SE)

    label_im = label(edges1)
    regions = regionprops(label_im)

    #ret, markers = cv2.connectedComponents(label_im)
    #markers = cv2.watershed(image, markers)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(image_gray)
    ax1.imshow(edges1)
    ax2.imshow(label_im)
    plt.show()
    # %%
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        #ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        #ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        #ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if (props.area_bbox>300):
            ax.plot(bx, by, '-b', linewidth=2.5)

    plt.show()


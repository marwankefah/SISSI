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
from utils.preprocess import illumination_correction
from alive_cells.test import nms
import pandas as pd


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
    cell_illumination_corrected = illumination_correction(image_gray)

    edges1 = feature.canny(cell_illumination_corrected, sigma=0.1)

    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges1 = edges1.astype(np.uint8)
    edges1 = cv2.morphologyEx(edges1*255, cv2.MORPH_CLOSE, SE)


    label_im = label(edges1, connectivity=2)
    regions = regionprops(label_im)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    boxeslist = {"cell_name": [], "x_min": [], "y_min": [], "x_max": [], "y_max": []}
    boxes_area = []

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
        if (props.area_bbox > 400):
            #boxeslist.append([bx, by])
            #ax.plot(bx, by, '-b', linewidth=2.5)
            boxeslist.append([minc, minr, maxc, maxr])

            boxeslist["x_min"].append(minc)
            boxeslist["y_min"].append(minr)
            boxeslist["x_max"].append(maxc)
            boxeslist["y_max"].append(maxr)
            boxes_area.append(props.area_bbox)

    bboxes = pd.DataFrame(boxeslist)
    bboxes_numpy = bboxes[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    bboxes_post_nms = np.asarray(nms(bboxes_numpy, np.ones_like(boxes_area), 0.15)[0])

    plt.show()

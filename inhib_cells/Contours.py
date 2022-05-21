import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
#from scipy import ndimage as ndi

#from skimage.segmentation import watershed
#from skimage.feature import peak_local_max
from skimage import feature
from skimage.color import rgb2gray
#from skimage import measure
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
dict_sum_counts = {}

for image, img_name in dead_images_raw:
    image_gray = rgb2gray(image)
    cell_illumination_corrected = illumination_correction(image_gray)

    edges1 = feature.canny(cell_illumination_corrected, sigma=0.1)

    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges1 = edges1.astype(np.uint8)
    edges1 = cv2.morphologyEx(edges1*255, cv2.MORPH_CLOSE, SE)


    label_im = label(edges1, connectivity=2)
    regions = regionprops(label_im)

    #fig, ax = plt.subplots()
    #ax.imshow(image, cmap=plt.cm.gray)

    boxeslist = {"cell_name": [], "x_min": [], "y_min": [], "x_max": [], "y_max": []}
    dict_sum_counts = {}
    boxes_area = []
    boxes = []
    img = image.copy()
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        minr, minc, maxr, maxc = props.bbox
        area = (maxc - minc) * (maxr - minr)
        dict_sum_counts[area] = dict_sum_counts.get(area, 0) + 1
        is_large_or_small = not (props.area_bbox < 400)
        if is_large_or_small and minc < maxc and minr < maxr:
            boxes.append([minc, minr, maxc, maxr])
            boxeslist["cell_name"].append("inhib")
            boxeslist["x_min"].append(minc)
            boxeslist["y_min"].append(minr)
            boxeslist["x_max"].append(maxc)
            boxeslist["y_max"].append(maxr)
            cv2.rectangle(img, (minr, minr),
                           (maxc, maxr), (255, 0, 0), 1)
            boxes_area.append(area)
        else:
            pass

    # plt.imshow(img)
    # plt.show()
    bboxes = pd.DataFrame(boxeslist)

    bboxes_numpy = bboxes[["x_min", "y_min", "x_max", "y_max"]].to_numpy()

    bboxes_post_nms = np.asarray(nms(bboxes_numpy, np.ones_like(boxes_area),0.15)[0])

    for (xmin, ymin, xmax, ymax) in bboxes_post_nms:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

    boxes = pd.DataFrame(bboxes_post_nms, columns=["x_min", "y_min", "x_max", "y_max"])
    boxes['cell_name'] = 'inhib'

    plt.imshow(image)
    plt.show()

plt.bar(dict_sum_counts.keys(), dict_sum_counts.values(), color='g')
plt.show()
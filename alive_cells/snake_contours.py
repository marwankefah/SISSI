import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
import skimage
from skimage import morphology
from pathlib import Path
import numpy as np
from skimage.filters import gaussian

import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation.active_contour_model import active_contour
from sklearn.cluster._mean_shift import estimate_bandwidth, MeanShift
from findmaxima2d import find_maxima, find_local_maxima  # Version that has been installed into python site-packages.

from utils.preprocess import fill_holes, imreconstruct, imposemin
import utils.cellpose_utils as cellpose_utils
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from utils.preprocess import illumination_correction, EGT_Segmentation, mask_overlay
from utils.seed_detection import seed_detection
from PIL import Image
from test import save_mask_png

cell_type = 'alive'

cell_path = "../data/chrisi/" + cell_type + "/"
data_dir = Path(
    "../data/weak_labels/" + cell_type + "/"
)

output_path = '../data/weak_labels_reduced/' + cell_type
if not os.path.exists(output_path):
    os.makedirs(output_path)


dead_images_raw = [
    [Image.open(str(img)), str(img).split('\\')[-1]] if str(img).split('\\')[-1] != '.gitignore' else [None, None] for
    img in data_dir.iterdir()
]

dead_images_raw.remove([None, None])
dict_sum_counts = {}
for image_mask, cell_mask_name in dead_images_raw:
    # print(cell_mask_name)
    cell_original_image = np.asarray(Image.open(os.path.join(cell_path, cell_mask_name.replace('_mask.png', '.jpg'))))

    label_mask_array = np.asarray(image_mask)
    label_mask_array = skimage.segmentation.expand_labels(label_mask_array, distance=3)

    cell_bbox_image = cell_original_image.copy()
    obj_ids = np.unique(label_mask_array)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    masks = label_mask_array == obj_ids[:, None, None]
    num_objs = len(obj_ids)
    boxes = []
    invalid_ids = []
    sum_counts = []
    for i in range(num_objs):
        # sum_counts.append(np.sum(masks[i]))
        object_mask_area = int(np.sum(masks[i]))
        dict_sum_counts[object_mask_area] = dict_sum_counts.get(object_mask_area, 0) + 1
        if object_mask_area <= 100 or object_mask_area >= 800:
            invalid_ids.append(i)

    masks = np.delete(masks, invalid_ids, axis=0)

    maski = np.zeros(shape=masks[0].shape, dtype=np.uint16)

    bbox = {"cell_name": [], "x_min": [],
            "y_min": [], "x_max": [], "y_max": []}

    for idx, mask in enumerate(masks):
        maski[mask == 1] = idx + 1

        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        cv2.rectangle(cell_bbox_image, (xmin - 2, ymin - 2),
                      (xmax + 2, ymax + 2), (255, 0, 0), 1)
        bbox["cell_name"].append("dead")

        bbox["x_min"].append(xmin - 5)
        bbox["y_min"].append(ymin - 5)
        bbox["x_max"].append(xmax + 5)
        bbox["y_max"].append(ymax + 5)

    outlines = cellpose_utils.masks_to_outlines(maski)
    outX, outY = np.nonzero(outlines)

    imgout = cell_original_image.copy()
    imgout[outX, outY] = np.array([255, 0, 0])  # pure red

    overlay = mask_overlay(cell_original_image, maski)

    # print(sorted(sum_counts)[0:4])

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    axes[0].imshow(imgout)
    axes[0].set_title('outlines of objects')

    axes[1].imshow(overlay)
    axes[1].set_title('Separated objects')

    axes[2].imshow(cell_bbox_image)
    axes[2].set_title('Separated objects Expanded')

    fig.tight_layout()
    # plt.savefig(os.path.join(output_path,cell_name))
    plt.show()
    save_mask_png(maski,cell_mask_name.replace('_mask.png', '.jpg'),output_path)

plt.bar(dict_sum_counts.keys(), dict_sum_counts.values(), color='g')
plt.show()
print(1)

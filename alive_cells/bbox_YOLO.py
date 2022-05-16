import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
from test import nms

cell_type = 'alive'
data_dir = Path(
    "../data/chrisi/" + cell_type + "/"
)
dead_images_raw = [
    [cv2.imread(str(img)), str(img).split('\\')[-1]] for img in data_dir.iterdir()
]
dead_images_raw.remove([None, '.gitignore'])

mask_path = Path(
    "../data/chrisi/weak_labels/alive/")

output_path = Path(
    "../data/chrisi/weak_labels_reduced_nms/alive")
dict_sum_counts = {}
for img, img_name in dead_images_raw:
    mask = np.asarray(Image.open(os.path.join(mask_path, img_name.replace('.jpg', '_mask.png'))))

    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]
    num_objs = len(obj_ids)
    boxes = []
    invalid_ids = []

    maski = np.zeros(shape=masks[0].shape, dtype=np.uint16)

    bbox = {"cell_name": [], "x_min": [],
            "y_min": [], "x_max": [], "y_max": []}
    boxes_area = []
    for idx, mask in enumerate(masks):
        maski[mask == 1] = idx + 1

        pos = np.where(mask)
        xmin = np.min(pos[1]) - 2
        xmax = np.max(pos[1]) + 2
        ymin = np.min(pos[0]) - 2
        ymax = np.max(pos[0]) + 2

        area = (xmax - xmin) * (ymax - ymin)
        dict_sum_counts[area] = dict_sum_counts.get(area, 0) + 1

        is_large_or_small = not (area < 400 or area > 1500)
        if is_large_or_small and xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0 and xmax < mask.shape[
            1] and ymax < \
                mask.shape[0]:
            boxes.append([xmin, ymin, xmax, ymax])
            # cv2.rectangle(img, (xmin, ymin),
            #               (xmax, ymax), (255, 0, 0), 1)
            bbox["cell_name"].append("alive")

            bbox["x_min"].append(xmin)
            bbox["y_min"].append(ymin)
            bbox["x_max"].append(xmax)
            bbox["y_max"].append(ymax)
            boxes_area.append(area)
        else:
            pass
            # print('invalid bbox found')

    bboxes = pd.DataFrame(bbox)

    bboxes_numpy = bboxes[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    bboxes_post_nms = np.asarray(nms(bboxes_numpy, np.ones_like(boxes_area), 0.15)[0])

    for (xmin, ymin, xmax, ymax) in bboxes_post_nms:
        cv2.rectangle(img, (xmin, ymin),
                      (xmax, ymax), (255, 0, 0), 1)

    boxes = pd.DataFrame(bboxes_post_nms, columns=["x_min", "y_min", "x_max", "y_max"])
    boxes['cell_name'] = 'alive'

    filename = img_name.split(".")[0]
    boxes[["cell_name", "x_min", "y_min", "x_max", "y_max"]].to_csv(
        str(output_path / Path(f"{filename}.txt")), sep=' ', header=None, index=None)

    plt.imshow(img)
    plt.show()

plt.bar(dict_sum_counts.keys(), dict_sum_counts.values(), color='g')
plt.show()

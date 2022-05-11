import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image

cell_type = 'test'
data_dir = Path(
    "../data/chrisi/" + cell_type + "/"
)
dead_images_raw = [
    [cv2.imread(str(img)), str(img).split('\\')[-1]] for img in data_dir.iterdir()
]
dead_images_raw.remove([None, '.gitignore'])


output_path = Path(
    "../data/weak_labels/alive/")

# output_path = Path(
#     "../data/test_labels/")

for img, img_name in dead_images_raw:
    mask = np.asarray(Image.open(os.path.join(output_path,img_name.replace('.jpg','_mask.png'))))

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

    for idx, mask in enumerate(masks):
        maski[mask == 1] = idx + 1

        pos = np.where(mask)
        xmin = np.min(pos[1]) - 2
        xmax = np.max(pos[1]) + 2
        ymin = np.min(pos[0]) - 2
        ymax = np.max(pos[0]) + 2

        if xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0 and xmax < mask.shape[1] and ymax < \
                mask.shape[0]:
            boxes.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(img, (xmin, ymin ),
                          (xmax, ymax), (255, 0, 0), 1)
            bbox["cell_name"].append("alive")

            bbox["x_min"].append(xmin)
            bbox["y_min"].append(ymin)
            bbox["x_max"].append(xmax)
            bbox["y_max"].append(ymax)
        else:
            pass
            # print('invalid bbox found')

    bboxes = pd.DataFrame(bbox)
    filename = img_name.split(".")[0]
    bboxes[["cell_name", "x_min", "y_min", "x_max", "y_max"]].to_csv(
        str(output_path/Path(f"{filename}.txt")), header=None, index=None)

    # plt.imshow(img)
    # plt.show()
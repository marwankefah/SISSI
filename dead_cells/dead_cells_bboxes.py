from types import CellType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


dead_data_path = Path("../data/chrisi/dead")
# dead_data_path = Path("raw/named_images_type/dead")


dead_images_raw = [
    [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE), str(img).split("/")[-1]]
    for img in dead_data_path.iterdir() if ".jpg" in str(img)
]

output_path = Path(
    "./data/chrisi/dead/output")
bbox_output_path = output_path/Path("bbox")
mask_output_path = output_path/Path("mask")

bbox_output_path.mkdir(exist_ok=True)
mask_output_path.mkdir(exist_ok=True)


for img, img_name in dead_images_raw:
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 10, minDist=25,
                               param1=100, param2=0.1, minRadius=7,
                               maxRadius=15
                               )

    output = img.copy()
    mask = np.zeros_like(img)
    if circles is not None:
        bbox = {"cell_name": [], "x_min": [],
                "y_min": [], "x_max": [], "y_max": []}

        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print((x, y, r))
            cv2.circle(output, (x, y), r+2, (0, 255, 0), 1)
            cv2.circle(mask, (x, y), r+2, (255, 255, 255), -1)

            cv2.rectangle(output, (x - r - 5, y - r - 5),
                          (x + r + 5, y + r + 5), (255, 0, 0), 2)
            bbox["cell_name"].append("dead")

            bbox["x_min"].append(x - r - 5)
            bbox["y_min"].append(y - r - 5)
            bbox["x_max"].append(x + r + 5)
            bbox["y_max"].append(y + r + 5)
        bboxes = pd.DataFrame(bbox)
        filename = img_name.split(".")[0]
        bboxes[["cell_name", "x_min", "y_min", "x_max", "y_max"]].to_csv(
            str(bbox_output_path/Path(f"{filename}.txt")), sep=' ', header=None, index=None)

        # cv2.imshow("output", np.hstack([img, output]))
        cv2.imwrite(str(output_path / img_name), np.hstack([img, output]))
        print(output_path / Path("mask") / img_name)
        cv2.imwrite(str(mask_output_path /
                    Path(f"{filename}.png")), mask)

        # cv2.waitKey(0)

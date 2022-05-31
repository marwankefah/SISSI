import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from settings import data_dir
from tqdm import tqdm


def get_bboxes_dead(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 10, minDist=25,
                               param1=100, param2=0.1, minRadius=7,
                               maxRadius=15
                               )

    output = img.copy()
    mask = np.zeros_like(img)

    if circles is not None:
        bbox = {"cell_type": [], "x_min": [],
                "y_min": [], "x_max": [], "y_max": []}

        circles = np.round(circles[0, :]).astype("int")
        i = 1

        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r+2, (0, 255, 0), 1)
            cv2.circle(mask, (x, y), r+2, (i, i, i), -1)

            cv2.rectangle(output, (x - r - 10, y - r - 10),
                          (x + r + 10, y + r + 10), (255, 0, 0), 2)
            bbox["cell_type"].append("dead")
            bbox["x_min"].append(max(x - r - 10, 0))
            bbox["y_min"].append(max(y - r - 10, 0))
            bbox["x_max"].append(min(x + r + 10, img.shape[1]-1))
            bbox["y_max"].append(min(y + r + 10, img.shape[0]-1))
            i += 1

        bboxes = pd.DataFrame(bbox)

        return bboxes[["cell_type", "x_min", "y_min", "x_max", "y_max"]]


if __name__ == "__main__":
    dead_data_path = Path("data/chrisi/dead")
    # dead_data_path = Path("raw/named_images_type/dead")

    dead_images_raw = [
        [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE), str(img).split("/")[-1]]
        for img in dead_data_path.iterdir() if ".jpg" in str(img)
    ]

    output_path = Path(
        "data/chrisi/output")
    output_path.mkdir(exist_ok=True)

    bbox_output_path = output_path/Path("bbox")
    mask_output_path = output_path/Path("mask")

    bbox_output_path.mkdir(exist_ok=True)
    mask_output_path.mkdir(exist_ok=True)
    for img, img_name in tqdm(dead_images_raw):
        filename = img_name.split(".")[0].split("/")[-1]
        bboxes, mask = get_bboxes_dead(img)
        bboxes.to_csv(
            str(bbox_output_path/Path(f"{filename}.txt")), sep=' ', header=None, index=None)
        cv2.imwrite(str(mask_output_path / Path(f"{filename}.png")), mask)

    print(f"Bounding boxes saved to: {str(bbox_output_path)}")

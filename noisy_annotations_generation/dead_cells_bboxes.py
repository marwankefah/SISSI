import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from random import randint
from settings import cell_lab_data_dir, noisy_bbox_output_dir
import matplotlib.pyplot as plt
from utils.preprocess import visualize



def get_bboxes_dead(img):
    output = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 10, minDist=25,
                               param1=100, param2=0.1, minRadius=7,
                               maxRadius=15
                               )

    mask = np.zeros_like(img)

    if circles is not None:
        bbox = {"cell_type": [], "x_min": [],
                "y_min": [], "x_max": [], "y_max": []}

        circles = np.round(circles[0, :]).astype("int")
        i = 1

        for (x, y, r) in circles:
            # cv2.circle(output, (x, y), r+2, (0, 255, 0), 1)
            cv2.circle(mask, (x, y), r + 2, (i, i, i), -1)

            cv2.rectangle(output, (x - r - 10, y - r - 10),
                          (x + r + 10, y + r + 10), (0, 0, 255), 1)
            bbox["cell_type"].append("dead")
            bbox["x_min"].append(max(x - r - 10, 0))
            bbox["y_min"].append(max(y - r - 10, 0))
            bbox["x_max"].append(min(x + r + 10, img.shape[1] - 1))
            bbox["y_max"].append(min(y + r + 10, img.shape[0] - 1))
            i += 1

        bboxes = pd.DataFrame(bbox)
        # cv2.imwrite(f"data/output/{randint(0, 200)}.png", output)
        return bboxes[["cell_type", "x_min", "y_min", "x_max", "y_max"]]


if __name__ == "__main__":
    cell_type = 'dead'
    is_visualize=True
    dead_data_path = cell_lab_data_dir / Path(cell_type)
    output_path = noisy_bbox_output_dir / Path(cell_type)
    bbox_output_path = output_path / Path("bbox")
    mask_output_path = output_path / Path("mask")
    output_path.mkdir(parents=True,exist_ok=True)
    bbox_output_path.mkdir(exist_ok=True)
    # mask_output_path.mkdir(exist_ok=True)

    dead_images_raw = [
        [cv2.imread(str(img)), str(img.stem)]
        for img in dead_data_path.iterdir() if ".jpg" in str(img)
    ]

    for img, img_name in tqdm(dead_images_raw):
        boxes = get_bboxes_dead(img)
        boxes.to_csv(
            str(bbox_output_path / Path(f"{img_name}.txt")), sep=' ', header=None, index=None)
        # cv2.imwrite(str(mask_output_path / Path(f"{filename}.png")), mask)
        if is_visualize:
            category_ids = boxes[["cell_type"]].values.flatten().tolist()
            category_id_to_name = {1: 'alive', 2: 'inhib', 3: 'dead'}
            inv_map = {v: k for k, v in category_id_to_name.items()}
            category_ids = list(map(inv_map.get, category_ids))
            out_img = visualize(img, boxes[["x_min", "y_min", "x_max", "y_max"]].to_numpy(), [1] * len(category_ids),
                                category_ids
                                , category_id_to_name)
            plt.imshow(out_img)
            plt.show()
    print(f"Bounding boxes saved to: {str(bbox_output_path)}")

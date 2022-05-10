import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

dead_data_path = Path("../data/chrisi/dead")

dead_images_raw = [
    [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE), str(img).split("/")[-1]]
    for img in dead_data_path.iterdir() if ".jpg" in str(img)
]

output_path = Path(
    "./data/chrisi/dead/output")

output_path.mkdir(exist_ok=True)


bbox = {"cell_name": [], "x_min": [], "y_min": [], "x_max": [], "y_max": []}
for img, img_name in dead_images_raw:
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 10, minDist=25,
                               param1=100, param2=0.1, minRadius=7,
                               maxRadius=15
                               )

    output = img.copy()
    print(circles.shape)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.rectangle(output, (x - r - 5, y - r - 5),
                          (x + r + 5, y + r + 5), (255, 0, 0), 2)
            bbox["cell_name"].append(img_name)
            bbox["x_min"].append(x - r - 5)
            bbox["y_min"].append(y - r - 5)
            bbox["x_max"].append(x + r + 5)
            bbox["y_max"].append(y + r + 5)

        cv2.imwrite(str(output_path / img_name), np.hstack([img, output]))

bboxes = pd.DataFrame(bbox)
bboxes.to_csv(str(output_path/Path("bboxes.txt")), header=None, index=None)

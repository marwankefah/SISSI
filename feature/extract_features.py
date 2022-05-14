"""

GLCM/haar-like features/
gabor filters/ SIFT (rotation invariant) 
and then followed by anyth (SVM/adaboost/KNN)

"""
# %%
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def crop_w_bboxes(image, bboxes):
    cropped = []
    for (index, row) in bboxes.iterrows():
        cropped.append((image[row["y_min"]:row["y_max"],
                       row["x_min"]:row["x_max"]], row["cell_type"]))
    return cropped


def get_all_cells(image_dir, bbox_dir):

    all_images_raw = [
        [cv2.imread(str(img)), str(img).split("/")[-1]]
        for img in image_dir.iterdir() if ".jpg" in str(img)
    ]
    col_list = ["cell_type", "x_min", "y_min", "x_max", "y_max"]
    cropped = []
    for img, img_name in all_images_raw:

        cell_num = img_name.split(".")[0]
        bbox_path = bbox_dir/Path(f"{cell_num}.txt")
        bbox = pd.read_csv(
            bbox_path, sep=" ", header=None, names=col_list)
        cropped.extend(crop_w_bboxes(image=img, bboxes=bbox))
    return cropped


def extract_features():
    all_cells_raw = get_all_cells(
        Path("/Users/manasikattel/cell-segmentation/raw/named_images_type/dead"), Path("/Users/manasikattel/cell-segmentation/data/chrisi/dead/output/bbox"))
    for (img, cell_type) in all_cells_raw:
        plt.imshow(img)
        plt.title(cell_type)
        plt.show()
    # all_cells_glcm = [(glcm(img), cell_type)
    #                   for (img, cell_type) in all_cells_raw]


# %%
extract_features()

# %%

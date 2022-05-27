# %%
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
from feature import glcm_features
from tqdm import tqdm
from settings import output_dir


def crop_w_bboxes(image, bboxes):
    cropped = []
    for (index, row) in bboxes.iterrows():
        cropped.append((index, row["cell_type"], image[row["y_min"]:row["y_max"],
                       row["x_min"]:row["x_max"]]))
    return cropped


def get_all_cells(image_dir, bbox_dir):
    all_images_raw = [
        [cv2.imread(str(img)), str(img).split("/")[-1]]
        for img in image_dir.rglob("*") if ".jpg" in str(img)
    ]
    col_list = ["cell_type", "x_min", "y_min", "x_max", "y_max"]
    cropped_all = []
    print("Cropping images")
    for img, img_name in tqdm(all_images_raw):

        cell_name = img_name.split(".")[0]
        bbox_path = bbox_dir/Path(f"{cell_name}.txt")
        bbox = pd.read_csv(
            bbox_path, sep=" ", header=None, names=col_list)
        cropped = crop_w_bboxes(image=img, bboxes=bbox)
        cropped = [(cell_name,) + elem for elem in cropped]

        cropped_all.extend(cropped)
    return cropped_all


def extract_features(save=True):
    all_cells_raw = get_all_cells(
        Path("data/chrisi"),
        Path("data/output/bbox"))
    [cv2.imwrite(str(Path("data/cropped") / Path(cell[2]) / Path(cell[2]) / Path(f"{cell[0]}_{cell[1]}_{cell[2]}.png")), cell[3])
     for cell in all_cells_raw]
    feature_dfs = []
    print("Extracting features")
    for (cell_name, index, cell_type, img) in tqdm(all_cells_raw):
        # if "inhib" in cell_type:
        # plt.imshow(img)
        # plt.title(cell_type + str(", ") + cell_name +
        #           str(", Cell no.: ") + str(index))
        # plt.show()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm_features_dict = glcm_features(gray_img)
        feature_df = pd.DataFrame(glcm_features_dict)

        feature_df["cell_name"] = cell_name
        feature_df["cell_no"] = index
        feature_df["cell_type"] = cell_type
        feature_dfs.append(feature_df)

    features = pd.concat(feature_dfs)
    features.to_csv(output_dir/Path("glcm_features.csv"))


if __name__ == "__main__":
    extract_features()

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
from cytotoxicity_classification.glcm import glcm_features
from tqdm import tqdm
from settings import noisy_annotations_generation_path, cell_lab_data_dir, noisy_bbox_output_dir


def crop_w_bboxes(image, bboxes):
    cropped = []
    for (index, row) in bboxes.iterrows():
        cropped.append((index, row["cell_type"], image[int(row["y_min"]):int(row["y_max"]),
                                                 int(row["x_min"]):int(row["x_max"])]))
    return cropped


def get_all_cells(image_dir, bbox_dir):
    all_images_raw = [
        [cv2.imread(str(img)), str(img.stem)]
        for img in image_dir.rglob("*") if ".jpg" in str(img)
    ]
    col_list = ["cell_type", "x_min", "y_min", "x_max", "y_max"]
    cropped_all = []
    print("Cropping images")
    for img, img_name in tqdm(all_images_raw):
        cell_name = Path(img_name).stem
        bbox_path = bbox_dir / Path(f"{cell_name}.txt")
        bbox = pd.read_csv(
            bbox_path, sep=" ", header=None, names=col_list)
        cropped = crop_w_bboxes(image=img, bboxes=bbox)
        cropped = [(cell_name,) + elem for elem in cropped]

        cropped_all.extend(cropped)
    return cropped_all


def extract_features(save=True):
    cell_types = ['alive', 'dead', 'inhib', 'test_labelled']

    for cell_type in cell_types:
        image_dir = cell_lab_data_dir / Path(cell_type)
        bbox_dir = noisy_annotations_generation_path / noisy_bbox_output_dir / Path(cell_type) / Path('bbox')
        all_cells_raw = get_all_cells(
            image_dir=image_dir,  # Path("data//test_labelled"),
            bbox_dir=bbox_dir)  # Path("data/annotations_test/test_labelled_3/"))
        cropped_dir = Path("cropped")

        if cell_type == 'test_labelled':
            cropped_dir = Path("cropped_test")

        for division in ['alive', 'dead', 'inhib']:
            cropped_dir_sub = cropped_dir / Path(division)/Path(division)
            cropped_dir_sub.mkdir(parents=True, exist_ok=True)

        [cv2.imwrite(str(cropped_dir / Path(cell[2]) /Path(cell[2])/ Path(f"{cell[0]}_{cell[1]}_{cell[2]}.png")),
                     cell[3])

         for cell in all_cells_raw]

    # feature_dfs = []
    # print("Extracting features")

    # for (cell_name, index, cell_type, img) in tqdm(all_cells_raw):
    # if "inhib" in cell_type:
    #     plt.imshow(img)
    #     plt.title(cell_type + str(", ") + cell_name +
    #               str(", Cell no.: ") + str(index))
    #     plt.show()
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     glcm_features_dict = glcm_features(gray_img)
    #     feature_df = pd.DataFrame(glcm_features_dict)
    #
    #     feature_df["cell_name"] = cell_name
    #     feature_df["cell_no"] = index
    #     feature_df["cell_type"] = cell_type
    #     feature_dfs.append(feature_df)
    #
    # features = pd.concat(feature_dfs)
    # features.to_csv(output_dir/Path("glcm_features.csv"))


if __name__ == "__main__":
    extract_features()

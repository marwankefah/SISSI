from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from skimage import feature
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from tqdm import tqdm
from settings import cell_lab_data_dir, noisy_bbox_output_dir
from utils.preprocess import illumination_correction, nms, visualize, visualize_bbox
import matplotlib.pyplot as plt


def get_bboxes_json(coco, image_id):
    # image object, doesn't contain image yet, just meta data: filename, size, ...
    img = coco.imgs[image_id]
    # image name for saving
    name = img["file_name"].split(".")
    name = name[0]

    cat_ids = coco.getCatIds()
    # ids of annotations (e.g. amount of ROIs=70)
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=False)
    # print("anns ids", anns_ids)

    # load annotations
    anns = coco.loadAnns(anns_ids)

    # count from 1 to len(anns), 'L'
    bbox = {"x_min": [],
            "y_min": [], "x_max": [], "y_max": [], "cell_type": []}
    for count, ann in enumerate(anns):
        cell_type_1 = coco.loadCats(ann['category_id'])[0]['name']
        if cell_type_1 == 'fibre':
            continue
        xmin = ann['bbox'][0]
        ymin = ann['bbox'][1]
        xmax = ann['bbox'][0] + ann['bbox'][2]
        ymax = ann['bbox'][1] + ann['bbox'][3]
        bbox["x_min"].append(xmin)
        bbox["y_min"].append(ymin)
        bbox["x_max"].append(xmax)
        bbox["y_max"].append(ymax)
        bbox["cell_type"].append(cell_type_1)

    boxes = pd.DataFrame(bbox)
    return boxes[["cell_type", "x_min", "y_min", "x_max", "y_max"]]


image_ids_dict = {
    'cell49': 2,
    'cell107': 3,
    'cell142': 4,
    'cell151': 5,
    'cell218': 6
}

if __name__ == "__main__":

    cell_type = 'test_labelled'
    is_visualize=True
    manual_annotation_dir= cell_lab_data_dir/Path('manual_test_annotation/labels_M.json')
    coco = COCO(manual_annotation_dir)

    test_data_path = cell_lab_data_dir / Path(cell_type)
    output_path = noisy_bbox_output_dir / Path(cell_type)
    bbox_output_path = output_path / Path("bbox")
    # mask_output_path = output_path / Path("mask")
    output_path.mkdir(parents=True, exist_ok=True)
    bbox_output_path.mkdir(exist_ok=True)
    # mask_output_path.mkdir(exist_ok=True)

    test_images_raw = [
        [cv2.imread(str(img)), str(img.stem)] for img in
        test_data_path.iterdir() if ".jpg" in str(img)]

    for image, img_name in tqdm(test_images_raw):
        boxes = get_bboxes_json(coco, image_ids_dict[img_name])
        boxes[["cell_type", "x_min", "y_min", "x_max", "y_max"]].to_csv(
            str(bbox_output_path / Path(f"{img_name}.txt")), sep=' ', header=None, index=None)
        if is_visualize:
            category_ids = boxes[["cell_type"]].values.flatten().tolist()
            category_id_to_name = {1: 'alive', 2: 'inhib', 3: 'dead'}
            inv_map = {v: k for k, v in category_id_to_name.items()}
            category_ids = list(map(inv_map.get, category_ids))
            out_img = visualize(image, boxes[["x_min", "y_min", "x_max", "y_max"]].to_numpy(), [1] * len(category_ids),
                                category_ids
                                , category_id_to_name)
            plt.imshow(out_img)
            plt.show()
    print(f"Bounding boxes saved to: {str(bbox_output_path)}")

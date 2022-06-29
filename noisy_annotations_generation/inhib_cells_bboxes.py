import numpy as np
import cv2
from skimage import feature
from skimage.color import rgb2gray
from pathlib import Path
from skimage.measure import label, regionprops
from utils.preprocess import illumination_correction, nms, visualize
import pandas as pd
from tqdm import tqdm
from settings import cell_lab_data_dir, noisy_bbox_output_dir
import matplotlib.pyplot as plt


def get_bboxes_inhib(image):
    image_gray = rgb2gray(image)
    cell_illumination_corrected = illumination_correction(image_gray)

    edges1 = feature.canny(cell_illumination_corrected, sigma=0.1)

    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges1 = edges1.astype(np.uint8)
    edges1 = cv2.morphologyEx(edges1 * 255, cv2.MORPH_CLOSE, SE)

    label_im = label(edges1, connectivity=2)
    regions = regionprops(label_im)

    boxeslist = {"cell_type": [], "x_min": [],
                 "y_min": [], "x_max": [], "y_max": []}
    dict_sum_counts = {}
    boxes_area = []
    boxes = []
    img = image.copy()
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        dict_sum_counts[props.area_bbox] = dict_sum_counts.get(
            props.area_bbox, 0) + 1
        is_large_or_small = not (
                props.area_bbox < 400 or props.area_bbox > 80000)
        if is_large_or_small and minc < maxc and minr < maxr:
            boxes.append([minc, minr, maxc, maxr])
            boxeslist["cell_type"].append("inhib")
            boxeslist["x_min"].append(minc)
            boxeslist["y_min"].append(minr)
            boxeslist["x_max"].append(maxc)
            boxeslist["y_max"].append(maxr)
            cv2.rectangle(img, (minr, minr),
                          (maxc, maxr), (255, 0, 0), 1)
            boxes_area.append(props.area_bbox)
        else:
            pass

    bboxes = pd.DataFrame(boxeslist)

    bboxes_numpy = bboxes[["x_min", "y_min", "x_max", "y_max"]].to_numpy()

    bboxes_post_nms = np.asarray(
        nms(bboxes_numpy, np.ones_like(boxes_area), 0.15)[0])

    boxes = pd.DataFrame(bboxes_post_nms, columns=[
        "x_min", "y_min", "x_max", "y_max"])
    boxes['cell_type'] = 'inhib'
    return boxes[["cell_type", "x_min", "y_min", "x_max", "y_max"]]


if __name__ == "__main__":

    cell_type = 'inhib'
    is_visualize = True
    inhib_data_path = cell_lab_data_dir / Path(cell_type)
    output_path = noisy_bbox_output_dir / Path(cell_type)
    bbox_output_path = output_path / Path("bbox")
    # mask_output_path = output_path / Path("mask")
    output_path.mkdir(parents=True, exist_ok=True)
    bbox_output_path.mkdir(exist_ok=True)
    # mask_output_path.mkdir(exist_ok=True)

    inhib_images_raw = [
        [cv2.imread(str(img)), str(img.stem)] for img in
        inhib_data_path.iterdir() if ".jpg" in str(img)]

    for image, img_name in tqdm(inhib_images_raw):
        boxes = get_bboxes_inhib(image)
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

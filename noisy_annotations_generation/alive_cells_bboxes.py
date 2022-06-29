import cv2
import skimage
import numpy as np
# Version that has been installed into python site-packages.
from findmaxima2d import find_maxima, find_local_maxima
import pandas as pd
from utils.preprocess import imreconstruct, imposemin, nms,visualize
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from pathlib import Path
from tqdm import tqdm
from settings import cell_lab_data_dir, noisy_bbox_output_dir
import matplotlib.pyplot as plt



def get_bboxes_alive(img):
    # gamma correction for the image
    lookUpTable = np.empty((1, 256), np.uint8)
    gamma = 1.5
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(img, lookUpTable)

    # clahe in the hsv space (cut on value)
    hsv_img = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(20, 20))
    v = clahe.apply(v)
    hsv_img = np.dstack((h, s, v))

    # return to the rgb space
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # getting external borders for watershed
    img1 = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    img2 = cv2.medianBlur(img1, 5)
    img3 = cv2.bilateralFilter(img2, 9, 75, 75)
    img4 = cv2.adaptiveThreshold(
        img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 0)
    img5 = skimage.img_as_ubyte(
        skimage.morphology.skeletonize(skimage.img_as_bool(img4)))

    # edge detection with sobel
    sobelx = cv2.Sobel(img1, cv2.CV_64F, 1, 0)  # Find x and y gradients
    sobely = cv2.Sobel(img1, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    # angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    magnitude = cv2.normalize(magnitude, magnitude, 0,
                              255, cv2.NORM_MINMAX).astype(np.uint8)

    # distancing blobs away from each others
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    eroded_image = cv2.morphologyEx(img1, cv2.MORPH_ERODE, SE)
    opened_by_reconstruction = imreconstruct(eroded_image, img1, radius=5)
    dilated_obr_image = cv2.morphologyEx(
        opened_by_reconstruction, cv2.MORPH_DILATE, SE)
    opened_closed_br_image = imreconstruct(cv2.bitwise_not(
        dilated_obr_image), cv2.bitwise_not(img1), radius=5)
    opened_closed_br_image = cv2.bitwise_not(opened_closed_br_image)
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening_foreground = cv2.morphologyEx(
        opened_closed_br_image, cv2.MORPH_OPEN, SE)
    eroded_foreground = cv2.morphologyEx(
        opening_foreground, cv2.MORPH_OPEN, SE)

    # getting sure foreground pixels with simple otsu threshold
    ret2, th2 = cv2.threshold(eroded_foreground, 0,
                              255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # combining the sure foreground with the external markers
    watershedInput = imposemin(
        magnitude / 255, cv2.bitwise_or(img5 // 255, th2 // 255))

    # getting the local maxima with ImageJ algorithm
    ntol = 50  # Noise Tolerance.
    img_data = rgb.copy()

    # Should your image be an RGB image.
    if img_data.shape.__len__() > 2:
        img_data = (np.sum(img_data, 2) / 3.0)

    local_max = find_local_maxima(img_data)
    y, x, regs = find_maxima(img_data, local_max, ntol)

    markers_imagej = np.zeros_like(img1)
    markers_imagej[y, x] = 255

    markers_imagej, _ = ndi.label(markers_imagej)

    mask = watershed(watershedInput, markers_imagej, mask=th2)
    return get_bbox_from_mask(mask)


def get_bbox_from_mask(mask):
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]
    boxes = []

    maski = np.zeros(shape=masks[0].shape, dtype=np.uint16)

    bbox = {"cell_type": [], "x_min": [],
            "y_min": [], "x_max": [], "y_max": []}
    boxes_area = []
    for idx, mask in enumerate(masks):
        maski[mask == 1] = idx + 1

        pos = np.where(mask)
        xmin = np.min(pos[1]) - 2
        xmax = np.max(pos[1]) + 2
        ymin = np.min(pos[0]) - 2
        ymax = np.max(pos[0]) + 2

        area = (xmax - xmin) * (ymax - ymin)
        # threshold were determined from the dataset after a barplot of all areas
        is_large_or_small = not (area < 400 or area > 1500)
        if is_large_or_small and xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0 and xmax < mask.shape[
            1] and ymax < \
                mask.shape[0]:
            boxes.append([xmin, ymin, xmax, ymax])
            bbox["cell_type"].append("alive")
            bbox["x_min"].append(xmin)
            bbox["y_min"].append(ymin)
            bbox["x_max"].append(xmax)
            bbox["y_max"].append(ymax)
            boxes_area.append(area)
        else:
            pass

    bboxes = pd.DataFrame(bbox)

    bboxes_numpy = bboxes[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    bboxes_post_nms = np.asarray(
        nms(bboxes_numpy, np.ones_like(boxes_area), 0.15)[0])

    boxes = pd.DataFrame(bboxes_post_nms, columns=[
        "x_min", "y_min", "x_max", "y_max"])
    boxes['cell_type'] = 'alive'

    return boxes[["cell_type", "x_min", "y_min", "x_max", "y_max"]]


if __name__ == "__main__":
    cell_type = 'alive'
    is_visualize=True
    alive_data_path = cell_lab_data_dir / Path(cell_type)
    output_path = noisy_bbox_output_dir/Path(cell_type)
    bbox_output_path = output_path / Path("bbox")
    # mask_output_path = output_path / Path("mask")
    output_path.mkdir(parents=True,exist_ok=True)
    bbox_output_path.mkdir(exist_ok=True)
    # mask_output_path.mkdir(exist_ok=True)

    alive_images_raw = [
        [cv2.imread(str(img), cv2.IMREAD_COLOR), str(img.stem)]
        for img in alive_data_path.iterdir() if ".jpg" in str(img)
    ]

    for img, img_name in tqdm(alive_images_raw):
        boxes = get_bboxes_alive(img)
        boxes.to_csv(
            str(bbox_output_path / Path(f"{img_name}.txt")), sep=' ', header=None, index=None)
        # cv2.imwrite(str(mask_output_path / Path(f"{img_name}.png")), mask)
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

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
import skimage
from skimage import morphology
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.cluster._mean_shift import estimate_bandwidth, MeanShift

from utils.preprocess import fill_holes, imreconstruct
import utils.cellpose_utils as cellpose_utils
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from utils.preprocess import illumination_correction, EGT_Segmentation, mask_overlay

cell_type = 'alive'
data_dir = Path(
    "../data/chrisi/" + cell_type + "/"
)
dead_images_raw = [
    [cv2.imread(str(img)), str(img).split('\\')[-1]] for img in data_dir.iterdir()
]

for image, cell_name in dead_images_raw:

    lookUpTable = np.empty((1, 256), np.uint8)
    gamma = 1.5
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(image, lookUpTable)

    # plt.imshow(res, cmap="gray")
    # plt.title('gamma ='+str(gamma))
    # plt.show()

    cell_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # plt.imshow(cell_gray, cmap="gray")
    # plt.show()

    cell_illumination_corrected = illumination_correction(cell_gray)

    # plt.imshow(cell_illumination_corrected, cmap="gray")
    # plt.show()
    hsv_img = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(20, 20))
    v = clahe.apply(v)

    hsv_img = np.dstack((h, s, v))

    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # Compute image gradient and percentiles

    img1 = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # filter to reduce noise
    img = cv2.medianBlur(rgb, 3)

    # img2 = cv2.medianBlur(img1, 5)
    # img3 = cv2.bilateralFilter(img2, 9, 75, 75)
    # img4 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 0)
    # img5 = skimage.img_as_ubyte(skimage.morphology.skeletonize(skimage.img_as_bool(img4)))
    # img6 = cv2.dilate(img5, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # cv2.imshow('image', img3)
    # cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(16, 16))
    cell_clahe = clahe.apply(cell_illumination_corrected.astype('uint8'))

    sobelx = cv2.Sobel(img1, cv2.CV_64F, 1, 0)  # Find x and y gradients
    sobely = cv2.Sobel(img1, cv2.CV_64F, 0, 1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    magnitude = cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    eroded_image = cv2.morphologyEx(img1, cv2.MORPH_ERODE, SE)
    opened_by_reconstruction = imreconstruct(eroded_image, img1, radius=5)
    dilated_obr_image = cv2.morphologyEx(opened_by_reconstruction, cv2.MORPH_DILATE, SE)

    opened_closed_br_image = imreconstruct(cv2.bitwise_not(dilated_obr_image), cv2.bitwise_not(img1), radius=5)

    opened_closed_br_image = cv2.bitwise_not(opened_closed_br_image)

    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    ret2, th2 = cv2.threshold(opened_closed_br_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = peak_local_max(opened_closed_br_image, min_distance=1, footprint=np.ones((20, 20)), labels=th2)
    #
    mask = np.zeros(img1.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    overlay = mask_overlay(opened_closed_br_image, markers)

    mask = np.zeros(opened_closed_br_image.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    # watershed output #Todo put distance in watershed or image??

    # imposemin(opened_closed_br_image,)

    labels = watershed(opened_closed_br_image, markers, mask=th2)

    # outlines for plotting from cellpose
    outlines = cellpose_utils.masks_to_outlines(labels)
    outX, outY = np.nonzero(outlines)
    imgout = opened_closed_br_image.copy()
    imgout = cv2.cvtColor(imgout, cv2.COLOR_GRAY2RGB)

    imgout[outX, outY] = np.array([255, 0, 0])  # pure red
    fig, axes = plt.subplots(ncols=5, figsize=(20, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(cell_clahe, cmap=plt.cm.gray)
    ax[0].set_title('cell clahe')

    ax[1].imshow(opened_closed_br_image, cmap=plt.cm.gray)
    ax[1].set_title('foreground')

    ax[2].imshow(imgout)
    ax[2].set_title('outlines of objects')

    overlay = mask_overlay(img1, labels)
    ax[3].imshow(overlay)
    ax[3].set_title('Separated objects')


    cv2.imshow('clahe', cell_clahe)
    cv2.imshow('mag', magnitude)
    cv2.imshow('angle', angle)
    cv2.imshow('image', img1)
    cv2.imshow('opening_by_reconstruction', opened_by_reconstruction)
    cv2.imshow('close_open_by_reconstruction', opened_closed_br_image)
    cv2.imshow('overaly', overlay)
    cv2.imshow('otsu', th2)
    cv2.waitKey(0)

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()

    # plt.savefig(os.path.join(output_path, cell_name))
    plt.show()

cv2.destroyAllWindows()

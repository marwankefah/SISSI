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
from findmaxima2d import find_maxima, find_local_maxima  # Version that has been installed into python site-packages.

from utils.preprocess import fill_holes, imreconstruct, imposemin
import utils.cellpose_utils as cellpose_utils
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from utils.preprocess import illumination_correction, EGT_Segmentation, mask_overlay
from utils.seed_detection import seed_detection

cell_type = 'alive'
data_dir = Path(
    "../data/chrisi/" + cell_type + "/"
)
dead_images_raw = [
    [cv2.imread(str(img)), str(img).split('\\')[-1]] for img in data_dir.iterdir()
]
dead_images_raw.remove([None, '.gitignore'])
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
    #
    # # filter to reduce noise
    # img = cv2.medianBlur(rgb, 3)

    img2 = cv2.medianBlur(img1, 5)
    img3 = cv2.bilateralFilter(img2, 9, 75, 75)
    img4 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 0)
    img5 = skimage.img_as_ubyte(skimage.morphology.skeletonize(skimage.img_as_bool(img4)))
    # img6 = cv2.dilate(img5, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # cv2.imshow('image', img5)
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
    opening_foreground = cv2.morphologyEx(opened_closed_br_image, cv2.MORPH_OPEN, SE)
    eroded_foreground = cv2.morphologyEx(opening_foreground, cv2.MORPH_OPEN, SE)

    ret2, th2 = cv2.threshold(eroded_foreground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    watershedInput = imposemin(magnitude / 255, cv2.bitwise_or(img5 // 255, th2 // 255))

    # TODO implement seed instead to get coords(markers)
    try:
        mask = seed_detection(cell_gray, th2, sigma0=10, alpha=0.03)
    except:
        raise NameError(f"Error for {cell_name} ")

    ntol = 50  # Noise Tolerance.
    img_data = np.array(image).astype(np.float64)
    img_data = rgb
    # Should your image be an RGB image.
    if img_data.shape.__len__() > 2:
        img_data = (np.sum(img_data, 2) / 3.0)

    # Finds the local maxima using mximum filter.
    local_max = find_local_maxima(img_data)

    y, x, regs = find_maxima(img_data, local_max, ntol)
    plt.figure(figsize=(16, 16))
    plt.imshow(image)
    plt.plot(x, y, 'r+')

    image_m = image.copy()
    markers_imagej = np.zeros_like(cell_gray)
    markers_imagej[y, x] = 255

    markers_imagej, _ = ndi.label(markers_imagej)

    # watershed_cv2 = cv2.watershed(image.astype(np.uint8), markers_imagej)

    # image_m[watershed_cv2 == -1] = [255, 0, 0]

    # plt.imshow(image_m)
    # plt.show()
    markers, _ = ndi.label(mask)
    overlay = mask_overlay(cell_clahe, markers)

    labels = watershed(watershedInput, markers_imagej, mask=th2)

    # outlines for plotting from cellpose
    outlines = cellpose_utils.masks_to_outlines(labels)
    outX, outY = np.nonzero(outlines)
    imgout = image.copy()
    # imgout = cv2.cvtColor(imgout, cv2.COLOR_GRAY2RGB)

    imgout[outX, outY] = np.array([255, 0, 0])  # pure red
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 6))
    axes[0, 0].imshow(cell_clahe, cmap=plt.cm.gray)
    axes[0, 0].set_title('cell clahe')

    axes[0, 1].imshow(opened_closed_br_image, cmap=plt.cm.gray)
    axes[0, 1].set_title('foreground')

    axes[0, 2].imshow(eroded_foreground, cmap=plt.cm.gray)
    axes[0, 2].set_title('eroded_foreground')

    axes[0, 3].imshow(watershedInput, cmap=plt.cm.gray)
    axes[0, 3].set_title('WaterShed input')

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('seed detection')

    axes[1, 1].imshow(imgout)
    axes[1, 1].set_title('outlines of objects')

    overlay = mask_overlay(image, labels)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Separated objects')

    fig.tight_layout()

    # plt.savefig(os.path.join(output_path, cell_name))
    plt.show()

cv2.destroyAllWindows()

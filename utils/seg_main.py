import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import utils.cellpose_utils as cellpose_utils
from skimage.segmentation import watershed
from utils.preprocess import illumination_correction, EGT_Segmentation, mask_overlay
from utils.seed_detection import seed_detection

cell_type = "inhib"
data_dir = Path("C:\Users\Enrique Almar\PycharmProjects\cell-segmentation\data\chrisi" + cell_type + "/")
# data_dir = Path("raw/named_images_type/" + cell_type + "/")


dead_images_raw = [
    [cv2.imread(str(img)), str(img).split("\\")[-1]]
    for img in data_dir.iterdir() if "cell2" in str(img)]

for image, cell_name in dead_images_raw:

    cell_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # plt.imshow(cell_gray, cmap="gray")
    # plt.show()

    cell_illumination_corrected = illumination_correction(cell_gray)

    # plt.imshow(cell_illumination_corrected, cmap="gray")
    # plt.show()

    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(16, 16))
    cell_clahe = clahe.apply(cell_illumination_corrected.astype("uint8"))

    # plt.imshow(cell_clahe, cmap="gray")
    # plt.show()

    S = EGT_Segmentation(cell_clahe, 8, 10, 2)
    # plt.imshow(S, cmap="gray")
    # plt.show()

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    S_complement = (~S).astype(np.uint8)
    S = S.astype(np.uint8)

    S = cv2.morphologyEx(S, cv2.MORPH_OPEN, SE)

    # plt.imshow(foreground, cmap='gray')
    # plt.show()

    foreground = cell_clahe.astype(np.uint8) * S

    background = cell_clahe.astype(np.uint8) * S_complement

    output_path = "foreground_seg/" + cell_type + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    distance = ndi.distance_transform_edt(foreground)

    # TODO implement seed instead to get coords(markers)
    try:
        mask = seed_detection(cell_gray, S, sigma0=10, alpha=0.03)
    except:
        raise NameError(f"Error for {cell_name} ")

    markers, _ = ndi.label(mask)
    # watershed output #Todo put distance in watershed or image??
    labels = watershed(cell_clahe, markers)

    # outlines for plotting from cellpose
    outlines = cellpose_utils.masks_to_outlines(labels)
    outX, outY = np.nonzero(outlines)
    imgout = foreground.copy()
    imgout = cv2.cvtColor(imgout, cv2.COLOR_GRAY2RGB)

    imgout[outX, outY] = np.array([255, 0, 0])  # pure red
    fig, axes = plt.subplots(ncols=5, figsize=(20, 6),
                             sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(cell_clahe, cmap=plt.cm.gray)
    ax[0].set_title("cell clahe")

    ax[1].imshow(foreground, cmap=plt.cm.gray)
    ax[1].set_title("foreground")

    ax[2].imshow(-distance, cmap=plt.cm.gray)
    ax[2].set_title("Distances")

    ax[3].imshow(imgout)
    ax[3].set_title("outlines of objects")

    overlay = mask_overlay(foreground, labels)
    ax[4].imshow(overlay)
    ax[4].set_title("Separated objects")

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()

    plt.savefig(os.path.join(output_path, cell_name))
    plt.show()

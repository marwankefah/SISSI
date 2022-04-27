import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import morphology
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from preprocess import fill_holes
import cellpose_utils as cellpose_utils
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from utils.preprocess import illumination_correction, EGT_Segmentation, mask_overlay

cell_type = 'inhib'
data_dir = Path(
    "../data/chrisi/" + cell_type + "/"
)
dead_images_raw = [
    [cv2.imread(str(img)), str(img).split('\\')[-1]] for img in data_dir.iterdir()
]

for image, cell_name in dead_images_raw:


    cell_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # plt.imshow(cell_gray, cmap="gray")
    # plt.show()

    cell_illumination_corrected = illumination_correction(cell_gray)

    # plt.imshow(cell_illumination_corrected, cmap="gray")
    # plt.show()

    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(16, 16))
    cell_clahe = clahe.apply(cell_illumination_corrected.astype('uint8'))

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

    output_path = 'foreground_seg/' + cell_type + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    distance = ndi.distance_transform_edt(S)
    #TODO implement seed instead to get coords(markers)
    coords = peak_local_max(distance, footprint=np.ones((10, 10)), labels=S)


    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    #watershed output
    labels = watershed(-distance, markers, mask=S)

    #outlines for plotting from cellpose
    outlines = cellpose_utils.masks_to_outlines(labels)
    outX, outY = np.nonzero(outlines)
    imgout = foreground.copy()
    imgout = cv2.cvtColor(imgout, cv2.COLOR_GRAY2RGB)

    imgout[outX, outY] = np.array([255, 0, 0])  # pure red
    fig, axes = plt.subplots(ncols=5, figsize=(20, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(cell_clahe, cmap=plt.cm.gray)
    ax[0].set_title('cell clahe')

    ax[1].imshow(foreground, cmap=plt.cm.gray)
    ax[1].set_title('foreground')

    ax[2].imshow(-distance, cmap=plt.cm.gray)
    ax[2].set_title('Distances')

    ax[3].imshow(imgout)
    ax[3].set_title('outlines of objects')

    overlay = mask_overlay(foreground, labels)
    ax[4].imshow(overlay)
    ax[4].set_title('Separated objects')


    for a in ax:
        a.set_axis_off()

    fig.tight_layout()

    plt.savefig(os.path.join(output_path, cell_name))
    plt.show()


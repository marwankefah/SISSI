import cv2
import numpy as np
from skimage import morphology
import warnings
import matplotlib.pyplot as plt


def illumination_correction(image):
    # Alternate filtering
    img = image.copy()
    for i in range(3, 100, 2):
        SE = cv2.getStructuringElement(cv2.MORPH_RECT, (i, 10))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, SE)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, SE)

    I_ilumination = image - 0.9 * img + np.mean(0.9 * img)
    return I_ilumination


def EGT_Segmentation(I, min_cell_size=1, upper_hole_size_bound=999999, manual_finetune=0):
    # this controls how far each increment of manual_finetune moves the percentile threshold
    greedy_step = 1

    # Compute image gradient and percentiles
    sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0)  # Find x and y gradients
    sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    S1 = magnitude[np.nonzero(magnitude)]
    ratio = (np.max(S1) - np.min(S1)) / 1000
    bins = np.arange(np.min(S1), np.max(S1), ratio, dtype=float)
    bins = np.concatenate((bins, [np.inf]))
    w, t = np.histogram(S1, bins)

    hist_mode_mean = round(np.mean(np.argsort(w)[::-1][0:3]))

    temp_hist = w / np.sum(w) * 100
    # # compute lower bound
    lower_bound = 3 * hist_mode_mean
    if lower_bound > np.size(temp_hist):
        warnings.warn('lower bound set to end of list.')
        lower_bound = np.size(temp_hist)
    #
    # # ensure that 75% of the pixels have been taken

    norm_hist = temp_hist / np.max(temp_hist)
    # idx should only be the first occur
    idx = np.argwhere(norm_hist[hist_mode_mean:] < 0.05)[0] + hist_mode_mean - 1

    upper_bound = max(idx, 18 * hist_mode_mean)

    # Compute the density metric
    if upper_bound > np.size(temp_hist):
        warnings.warn('upper bound set to end of list.')
        upper_bound = np.size(temp_hist)

    density_metric = np.sum(temp_hist[lower_bound:upper_bound])

    # # Fit a line between the 80th and the 40th percentiles from the plot above
    saturation1 = 3
    saturation2 = 42
    a = (95 - 40) / (saturation1 - saturation2)
    b = 95 - a * saturation1
    #
    # # Compute gradient threshold
    prct_value = round(a * density_metric + b)
    if prct_value > 98:
        prct_value = 98
    if prct_value < 25:
        prct_value = 25
    # # decrease or increase by a multiple of 5 percentile the manual input
    prct_value = prct_value - greedy_step * manual_finetune
    if prct_value > 100:
        prct_value = 100
    if prct_value < 1:
        prct_value = 1
    prct_value = prct_value / 100;

    threshold = percentile_computation(S1, prct_value)
    #
    # # Threshold the gradient image and perform some cleaning with morphological operations
    S = magnitude > threshold

    plt.imshow(S,cmap="gray")
    plt.show()

    #TODO +-
    S = fill_holes(S, upper_hole_size_bound)

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    S = cv2.morphologyEx(S, cv2.MORPH_ERODE, SE)

    S = cv2.morphologyEx(S, cv2.MORPH_ERODE, SE)

    S = morphology.remove_small_objects(S.astype(bool), min_size=min_cell_size, connectivity=8)

    return S


def percentile_computation(A, p):
    assert (p >= 0 and 1 >= p)

    B = A[~np.isnan(A)]

    if B.size == 0:
        T = np.NaN
        return T

    B = np.sort(B)

    indx = round(p * len(B) + 1)
    if indx<1:
        indx=1
    elif indx>len(B):
        indx=len(B)

    T = B[indx]

    # T = np.reshape(T, p.shape)

    return T

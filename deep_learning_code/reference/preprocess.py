import colorsys
import math

import cv2
import numpy as np
from skimage import morphology
import warnings
import matplotlib.pyplot as plt
from skimage import measure

def illumination_correction(image):
    # Alternate filtering
    img = image.copy()
    for i in range(3, 100, 2):
        SE = cv2.getStructuringElement(cv2.MORPH_RECT, (i, 10))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, SE)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, SE)

    I_ilumination = image - 0.9 * img + np.mean(0.9 * img)
    return I_ilumination


def EGT_Segmentation(I, min_cell_size=1, upper_hole_size_bound=np.inf, manual_finetune=0):
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


    S = fill_holes(S, upper_hole_size_bound)
    S = S.astype(np.uint8)
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

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
    if indx < 1:
        indx = 1
    elif indx > len(B):
        indx = len(B)

    T = B[indx]

    # T = np.reshape(T, p.shape)

    return T


def fill_holes(S, upper_bound):
    S = S.astype(bool)

    if math.isinf(upper_bound):
        all_labels = measure.label(~S, connectivity=2)

        values, counts = np.unique(all_labels, return_counts=True)
        counts.sort()
        upper_bound = counts[-1]
        if np.size(upper_bound) == 0:
            upper_bound = 0

    BWu = morphology.remove_small_objects(~(S.astype(bool)), min_size=upper_bound, connectivity=4)
    BWu = (~S).astype(np.uint8) - BWu.astype(np.uint8)
    S[BWu > 0] = 1

    return S


def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)
    Parameters
    ----------------
    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]
    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels
    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range
    Returns
    ----------------
    RGB: uint8, 3D array
        array of masks overlaid on grayscale image
    """
    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = rgb_to_hsv(colors)
    if img.ndim > 2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:, :, 2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max() + 1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            HSV[ipix[0], ipix[1], 0] = hues[n]
        else:
            HSV[ipix[0], ipix[1], 0] = colors[n, 0]
        HSV[ipix[0], ipix[1], 1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h,s,v), axis=-1)
    return hsv


def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 8):
    """Iteratively expand the markers white keeping them limited by the mask during each iteration.
    :param marker: Grayscale image where initial seed is white on black background.
    :param mask: Grayscale mask where the valid area is white on black background.
    :param radius Can be increased to improve expansion speed while causing decreased isolation from nearby areas.
    :returns A copy of the last expansion.
    Written By Semnodime.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded


def imposemin(img, minima):
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0
    mask = np.minimum((img + 1), marker)
    return morphology.reconstruction(marker, mask, method='erosion')

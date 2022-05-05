import numpy as np
from scipy.fft import fft2, ifft2
from skimage import feature
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import math
import click


def log_kernel(size, sigma) -> np.ndarray:
    """
    log_kernel Generate a laplacian of gaussian kernel

    _extended_summary_

    Parameters
    ----------
    size : int
        size of the kernel
    sigma : float
        variance for the Gaussian filter

    Returns
    -------
    np.ndarray
        Returns the kernel with size size*size
    """

    size = size if (size % 2 == 0) else size + 1
    log = [[0 for x in range(size)] for x in range(size)]
    sizeby2 = int(size / 2)
    for i in range(-sizeby2, sizeby2):
        for j in range(-sizeby2, sizeby2):
            log[i][j] = (
                (-1)
                / (np.pi * sigma**4)
                * (1 - (i**2 + j**2) / (2 * sigma**2))
                * np.exp(-(i**2 + j**2) / (2 * sigma**2))
            )
    return np.array(log)


def conv2_spec_symetric(x, h) -> np.ndarray:
    """
    conv2_spec_symetric fast spectral domain convolution


    Parameters
    ----------
    x : np.ndarray
        input image
    h : np.ndarray
        convolution mask

    Returns
    -------
    np.ndarray
        result image
    """
    n = math.floor(h.shape[0] / 2)
    m = math.floor(h.shape[1] / 2)
    x = np.pad(x, ((n, n), (m, m)), "symmetric")
    y = np.real(
        ifft2(fft2(x, (x.shape[0], x.shape[1]))
              * fft2(h, (x.shape[0], x.shape[1])))
    )
    y = np.roll(y, (-n, -m), axis=(0, 1))
    y = y[n: y.shape[0] - n, m: y.shape[1] - m]
    return y


def glogkernel(sigma_x, sigma_y, theta) -> np.ndarray:
    """
    Generate kernel for glog
    """
    N = math.ceil(2 * 3 * sigma_x)
    X, Y = np.meshgrid(
        np.linspace(0, N, N + 1) - N / 2, np.linspace(0, N, N + 1) - N / 2
    )
    a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (
        2 * sigma_y**2
    )
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + \
        np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (
        2 * sigma_y**2
    )

    D2Gxx = ((2 * a * X + 2 * b * Y) ** 2 - 2 * a) * np.exp(
        -(a * X**2 + 2 * b * X * Y + c * Y**2)
    )
    D2Gyy = ((2 * b * X + 2 * c * Y) ** 2 - 2 * c) * np.exp(
        -(a * X**2 + 2 * b * X * Y + c * Y**2)
    )
    Gaussian = np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
    LoG = (D2Gxx + D2Gyy) / np.sum(Gaussian)
    return LoG


def seed_detection(image, fg_mask, sigma0=10, alpha=0.03):
    """
    seed_detection Seed point extraction; uses the methodology below

    Reference:
    [1] Kong, H., Akakin, H.C., Sarma, S.E.: A generalized laplacian
    of gaussian filter for blob detection and its applications.
    IEEE Transactions on Cybernetics 43(6), 17191733 (2013).
    doi:10.1109/TSMCB.2012.2228639

    Parameters
    ----------
    image : np.ndarray
        input image
    fg_mask : np.ndarray
        binary foreground mask
    sigma0 : float
        roughly minimal cell perimeters
    alpha : float
        alpha parameter

    Returns
    -------
    np.ndarray
        Binary array with 1 for seed points
    """

    t = np.arange(np.log(sigma0), 3.5, 0.2)
    theta = np.pi / 4
    sigma = np.exp(t)

    # ====== LoG across all sigmas ======
    log_all = []
    for k in range(len(sigma)):
        gamma = 2
        sig = sigma[k]
        filter_size = 2 * math.ceil(3 * sig) + 1
        h = np.power(sig, gamma) * log_kernel(filter_size, sig)
        pom = conv2_spec_symetric(image, h)
        log_all.append(-1 * pom)

    zeta = []
    l_g = np.amax(log_all)
    for k in range(len(sigma)):
        s_i_l = np.amin(log_all[k])
        SS_i = (log_all[k] - s_i_l) / (l_g - s_i_l)
        zeta.append(np.sum(np.array(SS_i) > 0.6))

    best = zeta.index(max(zeta))  # %%
    sigma_x_min = sigma[max(best - 3, 1)]
    sigma_x_max = sigma[min(best + 3, len(sigma)-1)]
    sigma_x = np.arange(math.ceil(sigma_x_min), math.floor(sigma_x_max))

    R = np.zeros_like(image)
    Theta = np.linspace(0, np.pi - theta, round(np.pi / theta))

    for sx in sigma_x:
        sigma_y = np.arange(sigma0, sx - 1)
        for sy in sigma_y:
            for theta in Theta:
                nor = (1 + np.log(sx) ** alpha) * (1 + np.log(sy) ** alpha)

                LoG = glogkernel(sx, sy, theta)
                LoGn = nor * LoG
                tmp = conv2_spec_symetric(image, LoGn)
                R = R - tmp

    lm = feature.peak_local_max(R, indices=False)
    points = lm * fg_mask

    return points.astype("uint8")


@click.command()
@click.argument("image_path", type=str)
@click.argument("mask_path", type=str)
@click.argument("output_path", type=str)
@click.argument("sigma0", type=float, default=10)
@click.argument("alpha", type=float, default=0.03)
def plot_seeds(image_path, mask_path, output_path, sigma0, alpha):
    """
    plot_seeds plots seeds on image

    Parameters
    ----------
    image_path : str
        path of the image for which seeds are to be detected
    mask_path : str
        corresponding foreground mask
    output_path : str
        path of the seeds on main image plot to be saved
    sigma0 : float
        roughly minimal cell perimeters
    alpha : float
        alpha parameter
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fg_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    fg_mask = fg_mask > 127

    seed_points = seed_detection(image, fg_mask, sigma0=sigma0, alpha=alpha)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.morphologyEx(seed_points, cv2.MORPH_DILATE, kernel)

    plt.imshow(image, cmap="gray")
    plt.imshow(dilated, cmap="jet", alpha=0.5)
    plt.savefig(output_path)


if __name__ == "__main__":
    plot_seeds()

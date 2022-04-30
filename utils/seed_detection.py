import numpy as np
from scipy.fft import fft2, ifft2


def log_kernel(size, sigma) -> np.ndarray:
    """
    log_kernel Generate a laplacian of gaussian kernel

    _extended_summary_

    Parameters
    ----------
    size_2 : int
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
    n = np.floor(h.shape[0] / 2)
    m = np.floor(h.shape[1] / 2)
    x = np.pad(x, ((n, n), (m, m)), "symmetric")
    y = np.real(
        ifft2(fft2(x, (x.shape[0], x.shape[1])) * fft2(h, (x.shape[0], x.shape[1])))
    )
    y = np.roll(y, np.floor(-1 * h.shape / 2), axis=(0, 1))
    y = y[n : y.shape[0] - n - 1, m : y.shape[1] - m - 1]
    return y


def glogkernel(sigma_x, sigma_y, theta) -> np.ndarray:
    """
    Generate kernel for glog
    """
    N = np.ceil(2 * 3 * sigma_x)
    X, Y = np.meshgrid(
        np.linspace(0, N, N + 1) - N / 2, np.linspace(0, N, N + 1) - N / 2
    )
    a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (
        2 * sigma_y**2
    )
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
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

# from scipy.signal.signaltools import convolve2d
#
# from BaseHHO import *
# import os
# import cv2
# from skimage import morphology
# import matplotlib.pyplot as plt
# from HomomorphicFilter import *
# root_path='named_images_split/train'
# for i in os.listdir(root_path):
#     mask_grey = cv2.imread(os.path.join(root_path,i)),cv2.IMREAD_GRAYSCALE)
#     closing_kernel = np.ones((80, 80))
#     closedImage = morphology.closing(mask_grey, closing_kernel)
#     bottomHatImage = closedImage - mask_grey
#
#     homo_filter = HomomorphicFilter(a=0.75, b=1.25)
#     img_filtered = homo_filter.filter(I=bottomHatImage, filter_params=2)
#     medianImage = cv2.medianBlur(img_filtered, 3)
#
#     plt.imshow(img_filtered, cmap='gray')
#     plt.show()
#
#     bottomHatImage = mask_grey
#     # cv2_imshow(bottomHatImage)
#
#     # print(bottomHatImage.max(), bottomHatImage.min())
#
#
#     # lateral
#     lateralKernel = [[-0.025, -0.025, -0.025, -0.025, -0.025],
#                      [-0.025, -0.075, -0.075, -0.075, -0.025],
#                      [-0.025, -0.075, 1, -0.075, -0.025],
#                      [-0.025, -0.075, -0.075, -0.075, -0.025],
#                      [-0.025, -0.025, -0.025, -0.025, -0.025]]
#
#     Ienhanced = bottomHatImage + convolve2d(bottomHatImage, lateralKernel, mode='same')
#
#     Ienhanced = (Ienhanced - Ienhanced.min()) / Ienhanced.max()
#     Ienhanced *= 255
#     Ienhanced = (Ienhanced - Ienhanced.min()) / Ienhanced.max()
#     Ienhanced *= 255
#
#     # cv2_imshow(Ienhanced)
#
#     # MCET-HHO
#     histogram, bin_edges = np.histogram(Ienhanced, bins=256, range=(0, 255))
#     histogram = histogram / (Ienhanced.shape[0] * Ienhanced.shape[1])
#     histogram[0] = 0
#     hho = BaseHHO(30, 250, 3, histogram)
#     gbest, _ = hho.train()
#
#     # binarizing based on best threshold
#     gbest[gbest < 0] = 0
#     gbest[gbest > 255] = 255
#     th = sorted(gbest)
#     # print(th)
#     Ienhanced[Ienhanced > th[2]] = 0
#     Ienhanced[Ienhanced < th[0]] = 0
#     Ienhanced[Ienhanced != 0] = 255
#     plt.imshow(Ienhanced, cmap='gray')
#     plt.show()

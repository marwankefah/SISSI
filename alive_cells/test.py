import numpy as np
from PIL import Image
import os
def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def save_mask_png(labels, cell_name, output_path):
    mask = np.asarray(labels)
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    masks = mask == obj_ids[:, None, None]
    num_objs = len(obj_ids)
    boxes = []
    invalid_ids = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        # checking degenerated boxes or ugly boxes
        if xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0 and xmax <= mask[i].shape[1] and ymax <= \
                mask[i].shape[0]:
            boxes.append([xmin, ymin, xmax, ymax])
        else:
            invalid_ids.append(i)

    masks = np.delete(masks, invalid_ids, axis=0)

    maski = np.zeros(shape=masks[0].shape, dtype=np.uint16)
    for idx, mask in enumerate(masks):
        maski[mask == 1] = idx + 1

    Image.fromarray(maski).save(os.path.join(output_path, cell_name.split('.')[0] + '_mask.png'
                                                                                    ''))

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

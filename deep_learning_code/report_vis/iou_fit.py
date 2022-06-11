import pandas as pd
import numpy as np

import deep_learning_code.reference.utils as utils
import matplotlib.pyplot as plt

# iou = pd.read_csv('run-cell_mixed_batch_training_log_val-tag-info_AP_0.5_BOX.csv').Value.values
iou = pd.read_csv('mixed_final.csv').Value.values

threshold = 0.99

# iou_subset = iou[0:11 + 1]
# need_label_correction = utils.if_update(iou_subset, 11, n_epoch=100,
#                                         threshold=threshold)
# print(need_label_correction)

# x_data_fit = np.linspace(0, len(iou_subset) * 1 / 1, len(iou_subset))
# a, b, c = utils.fit(utils.curve_func, x_data_fit, iou_subset)
# y = [utils.curve_func(x, a, b, c) for x in range(len(iou_subset))]
# plt.plot(x_data_fit, y)
# plt.show()
#

iou_subset = iou
x_data_fit = np.linspace(0, len(iou_subset) * 1 / 1, len(iou_subset))
a, b, c = utils.fit(utils.curve_func, x_data_fit, iou_subset)
y = [utils.curve_func(x, a, b, c) for x in range(len(iou_subset))]
plt.plot(x_data_fit, y)

plt.plot(x_data_fit[5], y[5], 'ro')

plt.show()
# for i in range(0, len(iou)):
#     iou_subset = iou[0:i + 1]
#     print(iou_subset)
#     need_label_correction = utils.if_update(iou_subset, i, n_epoch=100,
#                                             threshold=threshold)
#     print(need_label_correction)
#
#     x_data_fit = np.linspace(0, len(iou_subset) * 1 / 1, len(iou_subset))
#     a, b, c = utils.fit(utils.curve_func, x_data_fit, iou_subset)
#     y = [utils.curve_func(x, a, b, c) for x in range(len(iou_subset))]
#     plt.plot(x_data_fit, y)
#     plt.show()

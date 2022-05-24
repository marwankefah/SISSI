import time

from configs.configs_inst_seg import Configs
from dataloaders.dataset import (ddsm_dataset_labelled, BaseFetaDataSets, RandomGenerator, ResizeTransform,
                                 TwoStreamBatchSampler)
from medpy import metric
from monai.data.utils import decollate_batch
from odach_our import oda
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from PIL import Image
from reference.engine import train_one_epoch, evaluate, test, save_check_point
from dataloaders.instance_seg_dataset import chrisi_dataset
import os
from tensorboardX import SummaryWriter
import reference.utils as utils

from dataloaders.instance_seg_dataset import cell_pose_dataset

from reference.engine import test_time_augmentation

configs = Configs('./configs/mask_rcnn_test.ini')
log_time = int(time.time())

snapshot_path = os.path.join(configs.model_path.split('.pth')[0], str(log_time), 'test')

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

# configs.train_writer = SummaryWriter(snapshot_path + '/log')
# configs.val_writer = SummaryWriter(snapshot_path + '/log_val')
configs.alive_writer = SummaryWriter(snapshot_path + '/log_alive')
configs.dead_writer = SummaryWriter(snapshot_path + '/log_dead')
configs.chrisi_test_writer = SummaryWriter(snapshot_path + '/log_chrisi_test')

db_chrisi_test = chrisi_dataset(configs.chrisi_cells_root_path, ['test_labelled'],
                                configs.val_detections_transforms)

db_train = cell_pose_dataset(configs.cell_pose_root_path, 'train', configs.train_transform)
db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)
db_chrisi_alive = chrisi_dataset(configs.chrisi_cells_root_path, ['alive'], configs.val_detections_transforms)
db_chrisi_dead = chrisi_dataset(configs.chrisi_cells_root_path, ['dead'], configs.val_detections_transforms)

chrisi_test_data_loader = torch.utils.data.DataLoader(
    db_chrisi_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)

db_chrisi_alive.sample_list = random.sample(db_chrisi_alive.sample_list, 20)
db_chrisi_dead.sample_list = random.sample(db_chrisi_dead.sample_list, 20)

alive_data_loader = torch.utils.data.DataLoader(
    db_chrisi_alive, batch_size=configs.labelled_bs, shuffle=True, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)
dead_data_loader = torch.utils.data.DataLoader(
    db_chrisi_dead, batch_size=configs.labelled_bs, shuffle=True, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)

configs.model.to(configs.device)

if not configs.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
torch.cuda.manual_seed(configs.seed)

medpy_dice_list = []

tta = [oda.HorizontalFlip(), oda.VerticalFlip()]
scale = [0.8, 0.9, 1, 1.1, 1.2]

# wrap model and tta
tta_model = oda.TTAWrapper(configs.model, tta, scale, nms="wbf", iou_thr=0.5, skip_box_thr=0.25, score_thresh=0.25)

with torch.no_grad():
    test_time_augmentation(configs, tta_model, alive_data_loader, configs.device,
                           writer=configs.alive_writer)

    test_time_augmentation(configs, tta_model, chrisi_test_data_loader, configs.device,
                           writer=configs.chrisi_test_writer)

    test_time_augmentation(configs, tta_model, dead_data_loader, configs.device,
                           writer=configs.dead_writer)

    # evaluate(configs, 0, chrisi_test_data_loader, device=configs.device, writer=configs.chrisi_test_writer,
    #          vis_every_iter=1)
    # evaluate(configs, 0, alive_data_loader, device=configs.device, writer=configs.alive_writer, vis_every_iter=5)
    # evaluate(configs, 0, dead_data_loader, device=configs.device, writer=configs.dead_writer, vis_every_iter=5)

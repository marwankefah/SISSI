import os

from torch.backends import cudnn
from torch.utils.data.dataset import ConcatDataset

import utils
import torch
import argparse
import logging
import os
import random
import shutil
import sys
import time
from monai.data.utils import decollate_batch
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.configs import *
from dataloaders.dataset import (ddsm_dataset_labelled, BaseFetaDataSets, RandomGenerator, ResizeTransform,
                                 TwoStreamBatchSampler)
import odach as oda
from torchvision.ops import boxes as box_ops
from configs.configs import Configs
from monai.visualize import plot_2d_or_3d_image
from medpy import metric
from PIL import ImageFile

from configs.configs_inst_seg import Configs

from dataloaders.instance_seg_dataset import PennFudanDataset, cell_pose_dataset, chrisi_dataset

import reference.utils as utils
from reference.engine import train_one_epoch, evaluate, test

from reference.engine import coco_evaluate, correct_labels, save_check_point, train_mixed_one_epoch


def train(configs, snapshot_path):
    configs.train_writer = SummaryWriter(snapshot_path + '/log')
    configs.val_writer = SummaryWriter(snapshot_path + '/log_val')
    configs.alive_writer = SummaryWriter(snapshot_path + '/log_alive')
    configs.dead_writer = SummaryWriter(snapshot_path + '/log_dead')

    configs.chrisi_test_writer = SummaryWriter(snapshot_path + '/log_chrisi_test')

    configs.chrisi_test_writer_tta = SummaryWriter(snapshot_path + '/log_chrisi_test_tta')

    configs.cell_pose_test_writer = SummaryWriter(snapshot_path + '/log_cell_pose_test')

    configs.model.to(configs.device)

    # db_train = cell_pose_dataset(configs.cell_pose_root_path, 'train', configs.train_transform)
    # db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)

    db_train = cell_pose_dataset(configs.cell_pose_root_path, 'train', configs.train_transform)
    db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)

    db_chrisi_alive = chrisi_dataset(configs.chrisi_cells_root_path, ['alive'], configs.val_detections_transforms,
                                     )
    db_chrisi_dead = chrisi_dataset(configs.chrisi_cells_root_path, ['dead'], configs.val_detections_transforms,
                                    )
    # db_chrisi_inhib = chrisi_dataset(configs.chrisi_cells_root_path, 'inhib', configs.train_detections_transforms)

    db_chrisi_test = chrisi_dataset(configs.chrisi_cells_root_path, ['test_labelled'],
                                    configs.val_detections_transforms)

    weak_label_chrisi_dataset = chrisi_dataset(configs.chrisi_cells_root_path, ['alive', 'dead'],
                                               configs.train_detections_transforms,
                                               cache_labels=True)

    weak_label_chrisi_dataset_val = chrisi_dataset(configs.chrisi_cells_root_path, ['alive', 'dead'],
                                                   configs.val_detections_transforms,
                                                   cache_labels=True)

    weak_label_chrisi_dataloader = torch.utils.data.DataLoader(
        weak_label_chrisi_dataset, batch_size=configs.labelled_bs, shuffle=True, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    cell_pose_train_dataloader = torch.utils.data.DataLoader(
        db_train, batch_size=configs.labelled_bs, shuffle=True, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    cell_pose_test_dataloader = torch.utils.data.DataLoader(
        db_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    initial_weak_labels_data_loader = torch.utils.data.DataLoader(
        weak_label_chrisi_dataset_val, batch_size=configs.labelled_bs * 4, shuffle=False,
        num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    db_chrisi_alive.sample_list = random.sample(db_chrisi_alive.sample_list, 20)
    db_chrisi_dead.sample_list = random.sample(db_chrisi_dead.sample_list, 20)

    chrisi_alive_data_loader = torch.utils.data.DataLoader(
        db_chrisi_alive, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    chrisi_dead_data_loader = torch.utils.data.DataLoader(
        db_chrisi_dead, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    chrisi_test_data_loader = torch.utils.data.DataLoader(
        db_chrisi_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    # coco_initial_labels = get_coco_api_from_dataset(weak_label_chrisi_dataset_val)

    # score_thresh
    # nms_thresh
    # detections_per_img

    # past_score_thresh = configs.model.roi_heads.score_thresh
    # past_detections_per_img = configs.model.roi_heads.detections_per_img
    # past_nms_thresh = configs.model.roi_heads.nms_thresh
    #
    # configs.model.roi_heads.score_thresh = 0.45
    # configs.model.roi_heads.detections_per_img = 250
    # configs.model.roi_heads.nms_thresh = 0.3

    # evaluate(configs, 0, chrisi_test_data_loader, configs.device, configs.chrisi_test_writer,
    #          vis_every_iter=1)
    #
    # configs.model.roi_heads.score_thresh = past_score_thresh
    # configs.model.roi_heads.detections_per_img = past_detections_per_img
    # configs.model.roi_heads.nms_thresh = past_nms_thresh

    # tta = [oda.HorizontalFlip(), oda.VerticalFlip(), oda.Rotate90Left(), oda.Multiply(0.9), oda.Multiply(1.1)]
    # tta_model = oda.TTAWrapper(configs.model, tta)

    configs.model.train()

    writer = configs.train_writer
    writer_val = configs.val_writer

    logging.info("{} iterations per epoch".format(len(weak_label_chrisi_dataloader)))

    iter_num = 0

    max_epoch = configs.max_iterations // len(weak_label_chrisi_dataloader) + 1
    iterator = tqdm(range(configs.start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:

        # train_iou, outputs_list_dict = evaluate(configs, epoch_num, initial_weak_labels_data_loader, configs.device,
        #                                         configs.val_writer,
        #                                         vis_every_iter=20)

        train_iou, outputs_list_dict = evaluate(configs, epoch_num, initial_weak_labels_data_loader, configs.device,
                                                configs.val_writer,
                                                vis_every_iter=20, use_tta=True)
        evaluate(configs, epoch_num, chrisi_alive_data_loader, configs.device,
                 configs.alive_writer, use_tta=True,
                 vis_every_iter=5)

        evaluate(configs, epoch_num, chrisi_dead_data_loader, configs.device,
                 configs.dead_writer, use_tta=True,
                 vis_every_iter=5)

        evaluate(configs, epoch_num, cell_pose_test_dataloader, device=configs.device,
                 writer=configs.cell_pose_test_writer)

        # evaluate chrisi testset
        AP_50_all, _ = evaluate(configs, epoch_num, chrisi_test_data_loader, configs.device, configs.chrisi_test_writer,
                                vis_every_iter=1)

        _, _ = evaluate(configs, epoch_num, chrisi_test_data_loader, configs.device,
                        configs.chrisi_test_writer_tta,
                        vis_every_iter=1, use_tta=True)

        save_check_point(configs, epoch_num, AP_50_all, snapshot_path)

        correct_labels(configs, weak_label_chrisi_dataset, outputs_list_dict, epoch_num, max_epoch)

        weak_label_chrisi_dataloader = torch.utils.data.DataLoader(
            weak_label_chrisi_dataset, batch_size=configs.labelled_bs, shuffle=True,
            num_workers=configs.num_workers,
            collate_fn=utils.collate_fn)

        train_mixed_one_epoch(configs, cell_pose_train_dataloader, weak_label_chrisi_dataloader, epoch_num, 20,
                              configs.train_writer)

        configs.lr_scheduler.step(AP_50_all)

        # configs.train_iou_values.append(train_iou)

        if iter_num >= configs.max_iterations:
            break
        configs.model.train()
        logging.info('{} epoch finished'.format(epoch_num + 1))

    writer.close()
    writer_val.close()
    return "Training Finished!"


if __name__ == "__main__":

    configs = Configs('./configs/mask_rcnn_mix.ini')

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

    log_time = int(time.time())

    snapshot_path = configs.model_output_path + "model/{}_labelled/{}/{}".format(
        configs.exp, configs.model_name, log_time)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(vars(configs)))
    train(configs, snapshot_path)

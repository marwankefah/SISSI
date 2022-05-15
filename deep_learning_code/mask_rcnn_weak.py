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

from deep_learning_code.reference.coco_utils import get_coco_api_from_dataset
from deep_learning_code.reference.engine import coco_evaluate


def train(configs, snapshot_path):
    configs.train_writer = SummaryWriter(snapshot_path + '/log')
    configs.val_writer = SummaryWriter(snapshot_path + '/log_val')
    configs.alive_writer = SummaryWriter(snapshot_path + '/log_alive')
    configs.chrisi_test_writer = SummaryWriter(snapshot_path + '/log_chrisi_test')

    configs.model.to(configs.device)

    # db_train = cell_pose_dataset(configs.cell_pose_root_path, 'train', configs.train_transform)
    # db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)

    db_chrisi_alive = chrisi_dataset(configs.chrisi_cells_root_path, ['alive'], configs.train_detections_transforms,
                                     )
    db_chrisi_dead = chrisi_dataset(configs.chrisi_cells_root_path, ['dead'], configs.train_detections_transforms,
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

    initial_weak_labels_data_loader = torch.utils.data.DataLoader(
        weak_label_chrisi_dataset_val, batch_size=configs.labelled_bs, shuffle=False, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    weak_label_chrisi_dataloader = torch.utils.data.DataLoader(
        weak_label_chrisi_dataset, batch_size=configs.labelled_bs, shuffle=True, num_workers=configs.num_workers,
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
    best_AP_50_all = configs.best_performance

    for epoch_num in iterator:

        # TODO return iou values in training
        train_one_epoch(configs, weak_label_chrisi_dataloader, epoch_num, print_freq=10,
                        writer=configs.train_writer)

        configs.lr_scheduler.step()

        train_iou, outputs_list_dict = evaluate(configs, epoch_num, initial_weak_labels_data_loader, configs.device,
                                                configs.val_writer,
                                                vis_every_iter=5)

        configs.train_iou_values.append(train_iou)

        # evaluate chrisi testset
        AP_50_all, _ = evaluate(configs, epoch_num, chrisi_test_data_loader, configs.device, configs.chrisi_test_writer,
                                vis_every_iter=1)
        # test(configs, epoch_num, alive_data_loader, configs.device, configs.alive_writer)  # AP iou 0.75--all bbox

        save_mode_path = os.path.join(snapshot_path,
                                      'epoch_{}_val_AP_50_all_{}.pth'.format(
                                          epoch_num, round(AP_50_all, 4)))
        logging.info('saving model with best performance {}'.format(AP_50_all))
        utils.save_on_master({
            'model': configs.model.state_dict(),
            'optimizer': configs.optimizer.state_dict(),
            'lr_scheduler': configs.lr_scheduler.state_dict(),
            'epoch': epoch_num,
            'best_performance': AP_50_all,
            'train_iou_values': configs.train_iou_values,
            'need_label_correction': configs.need_label_correction}, save_mode_path)

        if configs.label_correction:
            # if the flag is false, then check every time if it needs label correction
            # if it is true one time, it will always be true
            if not configs.need_label_correction:
                configs.need_label_correction = utils.if_update(configs.train_iou_values, epoch_num, n_epoch=max_epoch,
                                                                threshold=configs.label_correction_threshold)

            # it needs label correction, then output the label correction in a folder and reload it again
            # no large cache memory
            if configs.need_label_correction:
                logging.info('Label correction........')
                # we can easily put the output bboxes in the cached labels?
                for train_batch_output_dict in outputs_list_dict:
                    for idx, model_single_output in train_batch_output_dict.items():
                        # TODO that it is done correctly/do nMS and score threshold but just on function
                        # remove low scoring boxes
                        boxes = model_single_output['boxes']
                        scores = model_single_output['scores']
                        labels = model_single_output['labels']
                        inds = torch.where(scores > 0.5)[0]
                        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
                        # # remove empty boxes
                        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
                        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                        #
                        # non-maximum suppression, independently done per class
                        keep = box_ops.batched_nms(boxes, scores, labels, 0.35)
                        # keep only topk scoring predictions
                        keep = keep[: 200]
                        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                        # TODO boxes are for the image of 512x512
                        y_scale = weak_label_chrisi_dataset.img_orig_size[idx][0]
                        x_scale = weak_label_chrisi_dataset.img_orig_size[idx][1]
                        xmin, ymin, xmax, ymax = boxes.unbind(1)

                        xmin = xmin * x_scale
                        xmax = xmax * x_scale
                        ymin = ymin * y_scale
                        ymax = ymax * y_scale
                        boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)

                        if torch.numel(boxes) != 0:
                            weak_label_chrisi_dataset.sample_list[idx] = (
                                weak_label_chrisi_dataset.sample_list[idx][0], boxes.tolist())
                        else:
                            logging.info('image with id {} have no output'.format(idx))
                weak_label_chrisi_dataloader = torch.utils.data.DataLoader(
                    weak_label_chrisi_dataset, batch_size=configs.labelled_bs, shuffle=True,
                    num_workers=configs.num_workers,
                    collate_fn=utils.collate_fn)

        if iter_num >= configs.max_iterations:
            break
        configs.model.train()
        logging.info('{} epoch finished'.format(epoch_num + 1))

    writer.close()
    writer_val.close()
    return "Training Finished!"


if __name__ == "__main__":

    configs = Configs('./configs/mask_rcnn_weak.ini')

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

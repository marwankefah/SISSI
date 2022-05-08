import math
import sys
import time
from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
import cv2
import torch
import numpy as np
import logging
from reference.coco_utils import get_coco_api_from_dataset
from reference.coco_eval import CocoEvaluator
import reference.utils as utils
from collections import Counter

from reference.preprocess import mask_overlay

category_ids = [1]
# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {1: 'cell'}


def train_one_epoch(configs, data_loader, epoch, print_freq, writer):
    train_loss_list = []
    configs.model.train()

    header = 'Epoch: [{}]'.format(epoch)
    train_loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0,
                       'loss_rpn_box_reg': 0}
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(configs.optimizer, warmup_iters, warmup_factor)
    total_iter_per_epoch = len(data_loader)

    for iter_epoch, (images, targets) in enumerate(data_loader):
        images = list(image.to(configs.device) for image in images)

        targets = [{k: v.to(configs.device) for k, v in t.items()} for t in targets]

        loss_dict, outputs = configs.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        configs.optimizer.zero_grad()
        losses.backward()
        configs.optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        writer.add_scalar('info/lr', configs.optimizer.param_groups[0]["lr"], epoch)

        if iter_epoch % 20 == 0:
            # (epoch+1)*iter_epoch
            output_vis_to_tensorboard(images, targets, outputs, (epoch+1)*iter_epoch, writer)

        train_loss_dict = Counter(train_loss_dict) + Counter(loss_dict_reduced)

        loss_str = []
        for name, meter in loss_dict_reduced.items():
            loss_str.append(
                "{}: {}".format(name, str(round(float(meter), 5)))
            )

        logging.info('{} [{}/{}] loss:{} '.format(header, iter_epoch, total_iter_per_epoch,
                                                  round(loss_value, 4)) + "\t".join(loss_str))

        # TODO add images and predictions and masks to tensorboard
    train_losses_reduced = sum(loss for loss in train_loss_dict.values()) / total_iter_per_epoch
    loss_str = []
    for name, meter in train_loss_dict.items():
        writer.add_scalar('info/' + str(name), float(meter) / total_iter_per_epoch, epoch)
        loss_str.append(
            "{}: {}".format(name, str(round(float(meter) / total_iter_per_epoch, 5)))
        )

    writer.add_scalar('info/total_loss', train_losses_reduced, epoch)

    logging.info('{}  finished [{}/{}] lr: {} loss:{} '.format(header, iter_epoch, total_iter_per_epoch,
                                                               configs.optimizer.param_groups[0]["lr"],
                                                               train_losses_reduced) + "\t".join(loss_str))

    return


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    iou_types.append("segm")
    return iou_types


@torch.no_grad()
def evaluate(configs, epoch, data_loader, device, writer):
    n_threads = torch.get_num_threads()
    # FIXME (i need someone to fix me ) remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    configs.model.eval()
    header = 'Test: Epoch [{}]'.format(epoch)
    val_loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0,
                     'loss_rpn_box_reg': 0}
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(configs.model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    total_iter_per_epoch = len(data_loader)
    for iter_per_epoch, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets1 = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        with torch.no_grad():
            configs.model.rpn.training = True
            configs.model.roi_heads.training = True

            loss_dict, outputs = configs.model(images,targets1)

        if iter_per_epoch % 20 == 0:
            # (epoch+1)*iter_epoch
            output_vis_to_tensorboard(images, targets1, outputs, (epoch+1)*iter_per_epoch, writer)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        coco_evaluator.update(res)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        val_loss_dict = Counter(val_loss_dict) + Counter(loss_dict_reduced)

    # gather the stats from all processes

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    writer.add_scalar('info/AP_0.5_BOX', coco_evaluator.coco_eval['bbox'].stats[1], epoch)
    writer.add_scalar('info/AP_0.75_BOX', coco_evaluator.coco_eval['bbox'].stats[2], epoch)
    writer.add_scalar('info/AR__BOX', coco_evaluator.coco_eval['bbox'].stats[8], epoch)

    writer.add_scalar('info/AP_0.5_SEG', coco_evaluator.coco_eval['segm'].stats[1], epoch)
    writer.add_scalar('info/AP_0.75_SEG', coco_evaluator.coco_eval['segm'].stats[2], epoch)
    writer.add_scalar('info/AR__SEG', coco_evaluator.coco_eval['segm'].stats[8], epoch)

    # TODO fix loss
    val_losses_reduced = sum(loss for loss in val_loss_dict.values()) / total_iter_per_epoch
    loss_str = []
    for name, meter in val_loss_dict.items():
        writer.add_scalar('info/' + str(name), float(meter) / total_iter_per_epoch, epoch)
        loss_str.append(
            "{}: {}".format(name, str(round(float(meter) / total_iter_per_epoch, 5)))
        )

    writer.add_scalar('info/total_loss', val_losses_reduced, epoch)

    logging.info('{}  finished [{}/{}] loss:{} '.format(header, iter_per_epoch, total_iter_per_epoch,
                                                        val_losses_reduced) + "\t".join(loss_str))

    torch.set_num_threads(n_threads)
    return coco_evaluator

def test(configs, epoch, data_loader, device, writer):
    n_threads = torch.get_num_threads()
    # FIXME (i need someone to fix me ) remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    configs.model.eval()
    header = 'Testing: Epoch [{}]'.format(epoch)
    logging.info('Testing with no annotations')

    total_iter_per_epoch = len(data_loader)
    for iter_per_epoch, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        with torch.no_grad():
            loss_dict, outputs = configs.model(images)

        if iter_per_epoch % 10 == 0:
            # (epoch+1)*iter_epoch
            output_vis_to_tensorboard(images, outputs, outputs, (epoch+1)*iter_per_epoch, writer)

    torch.set_num_threads(n_threads)

    logging.info('{}  finished [{}/{}]'.format(header, iter_per_epoch, total_iter_per_epoch))
    return


def visualize_bbox(img, bbox, class_name, color=(150, 0, 0), thickness=1):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(150, 150, 150),
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    return img


def output_vis_to_tensorboard(images, targets1, outputs, iter_per_epoch, writer):
    img = images[0].detach().cpu().numpy()
    writer.add_image('image', img, iter_per_epoch)
    img_gt_boxes_channel_last = np.moveaxis(images[0].detach().cpu().numpy(), 0, -1)
    img_with_gt_boxes = visualize(img_gt_boxes_channel_last, targets1[0]['boxes'].detach().cpu().numpy(),
                                  targets1[0]['labels'].detach().cpu().numpy(), category_id_to_name)
    img_gt_boxes_channel_first = np.moveaxis(img_with_gt_boxes, -1, 0)
    writer.add_image('image_GT_boxes', img_gt_boxes_channel_first, iter_per_epoch)
    masks_binary = targets1[0]['masks'].detach().cpu().numpy()
    maski = np.zeros(shape=masks_binary[0].shape, dtype=np.uint16)
    for idx, mask in enumerate(masks_binary):
        maski[mask == 1] = idx + 1

    img_gt_overlay = mask_overlay(img_gt_boxes_channel_last, maski)
    img_gt_overlay_channel_first = np.moveaxis(img_gt_overlay, -1, 0)
    writer.add_image('image_GT_masks', img_gt_overlay_channel_first, iter_per_epoch)
    ##################################################################
    img_with_output_boxes = visualize(img_gt_boxes_channel_last, outputs[0]['boxes'].detach().cpu().numpy(),
                                      outputs[0]['labels'].detach().cpu().numpy(), category_id_to_name)
    img_gt_output_channel_first = np.moveaxis(img_with_output_boxes, -1, 0)
    writer.add_image('image_output_boxes', img_gt_output_channel_first, iter_per_epoch)
    masks_binary = outputs[0]['masks'].squeeze().detach().cpu().numpy()
    maski = np.zeros(shape=masks_binary[0].shape, dtype=np.uint16)
    for idx, mask in enumerate(masks_binary):
        maski[mask > 0.5] = idx + 1

    img_output_overlay = mask_overlay(img_gt_boxes_channel_last, maski)
    img_output_overlay_channel_first = np.moveaxis(img_output_overlay, -1, 0)
    writer.add_image('image_output_masks', img_output_overlay_channel_first, iter_per_epoch)

import math
import sys
import random
from odach_our import oda
import cv2
import torch
import numpy as np
import logging
from reference.coco_utils import get_coco_api_from_dataset
from reference.coco_eval import CocoEvaluator
import reference.utils as utils
from collections import Counter
from torchvision.ops import boxes as box_ops

from reference.preprocess import mask_overlay

category_ids = [1]
# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {1: 'alive', 2: 'inhib', 3: 'dead'}


def train_mixed_one_epoch(configs, data_loader_labeled, data_loader_weak, epoch, print_freq, writer):
    configs.model.train()

    header = 'Epoch: [{}]'.format(epoch)
    train_loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0,
                       'loss_rpn_box_reg': 0}
    total_iter_per_epoch = len(data_loader_weak)
    labeled_data_loader_iter = iter(data_loader_labeled)
    weak_labeled_data_loader_iter = iter(data_loader_weak)

    for i_batch in range(0, total_iter_per_epoch):
        sampled_labelled_batch = labeled_data_loader_iter.next()
        sampled_weak_labelled_batch = weak_labeled_data_loader_iter.next()
        # images, targets, cell_names = images

        input = sampled_labelled_batch[0] + sampled_weak_labelled_batch[0]
        label = sampled_labelled_batch[1] + sampled_weak_labelled_batch[1]

        loss_dict_reduced, loss_value = train_one_iter(configs, i_batch, epoch, input, label, writer)

        writer.add_scalar('info/lr', configs.optimizer.param_groups[0]["lr"], epoch)

        train_loss_dict = Counter(train_loss_dict) + Counter(loss_dict_reduced)

        loss_str = []
        for name, meter in loss_dict_reduced.items():
            loss_str.append(
                "{}: {}".format(name, str(round(float(meter), 5)))
            )

        logging.info('{} [{}/{}] loss:{} '.format(header, i_batch, total_iter_per_epoch,
                                                  round(loss_value, 4)) + "\t".join(loss_str))

    train_losses_reduced = sum(loss for loss in train_loss_dict.values()) / total_iter_per_epoch
    loss_str = []
    for name, meter in train_loss_dict.items():
        writer.add_scalar('info/' + str(name), float(meter) / total_iter_per_epoch, epoch)
        loss_str.append(
            "{}: {}".format(name, str(round(float(meter) / total_iter_per_epoch, 5)))
        )

    writer.add_scalar('info/total_loss', train_losses_reduced, epoch)

    logging.info('{}  finished [{}/{}] lr: {} loss:{} '.format(header, i_batch, total_iter_per_epoch,
                                                               configs.optimizer.param_groups[0]["lr"],
                                                               train_losses_reduced) + "\t".join(loss_str))


def train_one_epoch(configs, data_loader, epoch, print_freq, writer):
    configs.model.train()

    header = 'Epoch: [{}]'.format(epoch)
    train_loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0,
                       'loss_rpn_box_reg': 0}
    lr_scheduler = None
    # outputs_list_dict=[]
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(configs.optimizer, warmup_iters, warmup_factor)
    total_iter_per_epoch = len(data_loader)

    for iter_epoch, (images, targets, cell_names) in enumerate(data_loader):

        loss_dict_reduced, loss_value = train_one_iter(configs, iter_epoch, epoch, images, targets, writer)

        if lr_scheduler is not None:
            lr_scheduler.step()
        writer.add_scalar('info/lr', configs.optimizer.param_groups[0]["lr"], epoch)

        train_loss_dict = Counter(train_loss_dict) + Counter(loss_dict_reduced)

        loss_str = []
        for name, meter in loss_dict_reduced.items():
            loss_str.append(
                "{}: {}".format(name, str(round(float(meter), 5)))
            )

        logging.info('{} [{}/{}] loss:{} '.format(header, iter_epoch, total_iter_per_epoch,
                                                  round(loss_value, 4)) + "\t".join(loss_str))

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


def train_one_iter(configs, iter_epoch, epoch, images, targets, writer):
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

    if iter_epoch % 20 == 0:
        output_vis_to_tensorboard(images, targets, outputs, (iter_epoch + epoch * 200), writer, configs.train_mask)
    return loss_dict_reduced, loss_value


def _get_iou_types(model, has_mask):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if has_mask:
        iou_types.append("segm")
    return iou_types


@torch.no_grad()
def evaluate(configs, epoch, data_loader, device, writer, vis_every_iter=20, use_tta=False, return_metrics=False):
    n_threads = torch.get_num_threads()
    # FIXME (i need someone to fix me ) remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    configs.model.eval()
    configs.model.rpn.training = True
    configs.model.roi_heads.training = True

    if use_tta:
        tta = [oda.HorizontalFlip(), oda.VerticalFlip()]
        scale = [0.8, 0.9, 1, 1.1, 1.2]
        model = oda.TTAWrapper(configs.model, tta, scale, nms="wbf", iou_thr=0.5, skip_box_thr=0.25,
                               score_thresh=configs.label_corr_score_thresh)
    else:
        model = configs.model

    header = 'Test: Epoch [{}]'.format(epoch)
    val_loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0,
                     'loss_rpn_box_reg': 0}
    outputs_list_dict = []

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(configs.model, configs.train_mask)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    total_iter_per_epoch = len(data_loader)
    for iter_per_epoch, (images, targets, cell_names) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets1 = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        with torch.no_grad():
            loss_dict, outputs = model(images, targets1)

        if iter_per_epoch % vis_every_iter == 0:
            output_vis_to_tensorboard(images, targets1, outputs, (iter_per_epoch + epoch * 200), writer,
                                      configs.train_mask)
            logging.info('Evaluation [{}/{}] '.format(iter_per_epoch, total_iter_per_epoch))

        for o, t in zip(outputs, targets1):
            o.update({'image_size': t['image_size']})

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        outputs_list_dict.append(res)

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
    if configs.train_mask:
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
    if return_metrics:
        return coco_evaluator.coco_eval['bbox']

    return coco_evaluator.coco_eval['bbox'].stats[1], outputs_list_dict, val_losses_reduced


def test(configs, epoch, data_loader, device, writer):
    # n_threads = torch.get_num_threads()
    # FIXME (i need someone to fix me ) remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    configs.model.eval()
    header = 'Testing: Epoch [{}]'.format(epoch)
    logging.info('Testing with no annotations')

    total_iter_per_epoch = len(data_loader)
    for iter_per_epoch, (images, targets, cell_names) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets1 = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        with torch.no_grad():
            loss_dict, outputs = configs.model(images)

        if iter_per_epoch % 10 == 0:
            output_vis_to_tensorboard(images, targets1, outputs, (iter_per_epoch + epoch * 200), writer,
                                      configs.train_mask)

    # torch.set_num_threads(n_threads)

    logging.info('{}  finished [{}/{}]'.format(header, iter_per_epoch, total_iter_per_epoch))
    return


def test_time_augmentation(configs, tta_model, data_loader, device, writer):
    # Execute TTA!
    configs.model.eval()
    for iter_per_epoch, (images, targets, cell_names) in enumerate(data_loader):
        targets1 = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = list(img.to(device) for img in images)

        outputs = tta_model(torch.stack(images).cuda())

        output_vis_to_tensorboard(images, targets1, outputs, iter_per_epoch, writer,
                                  configs.train_mask)


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


def visualize(image, bboxes, scores, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id, score in zip(bboxes, category_ids, scores):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, '{} {:.2f}'.format(class_name, score))
    return img


def output_vis_to_tensorboard(images, targets1, outputs, iter_per_epoch, writer, train_mask):
    random_val_visualize_int = random.randint(0, len(images) - 1)
    img = images[random_val_visualize_int].detach().cpu().numpy()
    writer.add_image('image', img, iter_per_epoch)
    img_gt_boxes_channel_last = np.moveaxis(img, 0, -1)
    img_with_gt_boxes = visualize(img_gt_boxes_channel_last,
                                  targets1[random_val_visualize_int]['boxes'].detach().cpu().numpy(),
                                  targets1[random_val_visualize_int]['labels'].detach().cpu().numpy(),
                                  targets1[random_val_visualize_int]['labels'].detach().cpu().numpy(),
                                  category_id_to_name)
    img_gt_boxes_channel_first = np.moveaxis(img_with_gt_boxes, -1, 0)
    writer.add_image('image_GT_boxes', img_gt_boxes_channel_first, iter_per_epoch)
    if train_mask:
        masks_binary = targets1[random_val_visualize_int]['masks'].detach().cpu().numpy()
        maski = np.zeros(shape=masks_binary[0].shape, dtype=np.uint16)
        for idx, mask in enumerate(masks_binary):
            maski[mask == 1] = idx + 1

        img_gt_overlay = mask_overlay(img_gt_boxes_channel_last, maski)
        img_gt_overlay_channel_first = np.moveaxis(img_gt_overlay, -1, 0)
        writer.add_image('image_GT_masks', img_gt_overlay_channel_first, iter_per_epoch)
    ##################################################################
    img_with_output_boxes = visualize(img_gt_boxes_channel_last,
                                      outputs[random_val_visualize_int]['boxes'].detach().cpu().numpy(),
                                      outputs[random_val_visualize_int]['scores'].detach().cpu().numpy(),
                                      outputs[random_val_visualize_int]['labels'].detach().cpu().numpy(),
                                      category_id_to_name)
    img_gt_output_channel_first = np.moveaxis(img_with_output_boxes, -1, 0)
    writer.add_image('image_output_boxes', img_gt_output_channel_first, iter_per_epoch)
    if train_mask:
        masks_binary = outputs[random_val_visualize_int]['masks'].squeeze().detach().cpu().numpy()
        maski = np.zeros(shape=masks_binary[0].shape, dtype=np.uint16)
        for idx, mask in enumerate(masks_binary):
            maski[mask > 0.5] = idx + 1
        img_output_overlay = mask_overlay(img_gt_boxes_channel_last, maski)
        img_output_overlay_channel_first = np.moveaxis(img_output_overlay, -1, 0)
        writer.add_image('image_output_masks', img_output_overlay_channel_first, iter_per_epoch)


# works with batch size =1
def coco_evaluate(outputs_list_dict, coco, epoch, writer, train_mask=False):
    n_threads = torch.get_num_threads()
    # FIXME (i need someone to fix me ) remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)

    header = 'Training Evaluation: Epoch [{}]'.format(epoch)
    iou_types = ["bbox"]
    if train_mask:
        iou_types.append('segm')

    coco_evaluator = CocoEvaluator(coco, iou_types)

    for res in outputs_list_dict:
        # torch.cuda.synchronize()
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    writer.add_scalar('info/AP_0.5_BOX', coco_evaluator.coco_eval['bbox'].stats[1], epoch)
    writer.add_scalar('info/AP_0.75_BOX', coco_evaluator.coco_eval['bbox'].stats[2], epoch)
    writer.add_scalar('info/AR__BOX', coco_evaluator.coco_eval['bbox'].stats[8], epoch)
    if train_mask:
        writer.add_scalar('info/AP_0.5_SEG', coco_evaluator.coco_eval['segm'].stats[1], epoch)
        writer.add_scalar('info/AP_0.75_SEG', coco_evaluator.coco_eval['segm'].stats[2], epoch)
        writer.add_scalar('info/AR__SEG', coco_evaluator.coco_eval['segm'].stats[8], epoch)

    logging.info('{}  finished AP 0.5:{} '.format(header, coco_evaluator.coco_eval['bbox'].stats[1]))

    # torch.set_num_threads(n_threads)
    return coco_evaluator.coco_eval['bbox'].stats[1]


def correct_labels(configs, weak_label_chrisi_dataset, outputs_list_dict, epoch_num, max_epoch):
    if configs.label_correction:
        # it needs label correction, then output the label correction in a folder and reload it again
        # no large cache memory
        if configs.need_label_correction:
            logging.info('Label correction........')
            # we can easily put the output bboxes in the cached labels?
            for train_batch_output_dict in outputs_list_dict:
                for idx, model_single_output in train_batch_output_dict.items():
                    # remove low scoring boxes
                    boxes = model_single_output['boxes']
                    scores = model_single_output['scores']
                    labels = model_single_output['labels']
                    inds = torch.where(scores >= configs.label_corr_score_thresh)[0]
                    boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

                    # # remove empty boxes
                    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                    #
                    # non-maximum suppression, independently done per class
                    keep = box_ops.batched_nms(boxes, scores, labels, 0.2)
                    # keep only topk scoring predictions
                    keep = keep[: 200]
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                    y_scale = (model_single_output['image_size'][0] - 1) / configs.patch_size[0]
                    x_scale = (model_single_output['image_size'][1] - 1) / configs.patch_size[1]
                    xmin, ymin, xmax, ymax = boxes.unbind(1)

                    xmin = xmin * x_scale
                    xmax = xmax * x_scale
                    ymin = ymin * y_scale
                    ymax = ymax * y_scale
                    boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
                    # TODO add label smoothing also?
                    if torch.numel(boxes) != 0:
                        weak_label_chrisi_dataset.sample_list[idx] = (
                            weak_label_chrisi_dataset.sample_list[idx][0], boxes.tolist())
                    else:
                        logging.info('image with id {} have no output'.format(idx))

        # if the flag is false, then check every time if it needs label correction
        # if it is true one time, it will always be true
        else:
            configs.need_label_correction = utils.if_update(configs.train_iou_values, epoch_num, n_epoch=max_epoch,
                                                            threshold=configs.label_correction_threshold)


import os
import pandas as pd


def output_data_set_labels(configs, weak_label_chrisi_dataset, outputs_list_dict, output_folder):
    for train_batch_output_dict in outputs_list_dict:
        for idx, model_single_output in train_batch_output_dict.items():
            # remove low scoring boxes
            boxes = model_single_output['boxes']
            scores = model_single_output['scores']
            labels = model_single_output['labels']
            inds = torch.where(scores >= configs.label_corr_score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            #
            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, 0.2)
            # keep only topk scoring predictions
            keep = keep[: 200]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            y_scale = (model_single_output['image_size'][0] - 1) / configs.patch_size[0]
            x_scale = (model_single_output['image_size'][1] - 1) / configs.patch_size[1]
            xmin, ymin, xmax, ymax = boxes.unbind(1)

            xmin = xmin * x_scale
            xmax = xmax * x_scale
            ymin = ymin * y_scale
            ymax = ymax * y_scale

            # TODO cast as int
            boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1).type(torch.int32)

            cell_type_name = weak_label_chrisi_dataset.images_path[idx].split('\\')
            filename = cell_type_name[1].split(".")[0]

            if torch.numel(boxes) != 0:
                boxes_list = boxes.tolist()
                boxes = pd.DataFrame(boxes_list, columns=["x_min", "y_min", "x_max", "y_max"])
                boxes['cell_name'] = cell_type_name[0]
                boxes[["cell_name", "x_min", "y_min", "x_max", "y_max"]].to_csv(
                    os.path.join(output_folder, f"{filename}.txt"), sep=' ', header=None, index=None)

            else:
                # TODO open empty .txt file
                logging.info('image with id {} have no output'.format(idx))
                open(os.path.join(output_folder, f"{filename}.txt"), 'a').close()


def save_check_point(configs, epoch_num, perforamnce, snapshot_path):
    save_mode_path = os.path.join(snapshot_path,
                                  'epoch_{}_perforamnce_{}.pth'.format(
                                      epoch_num, round(float(perforamnce.detach().cpu().numpy()), 4)))
    logging.info('saving model with best performance {}'.format(perforamnce))
    utils.save_on_master({
        'model': configs.model.state_dict(),
        'optimizer': configs.optimizer.state_dict(),
        'lr_scheduler': configs.lr_scheduler.state_dict(),
        'epoch': epoch_num,
        'best_performance': perforamnce,
        'train_iou_values': configs.train_iou_values,
        'need_label_correction': configs.need_label_correction}, save_mode_path)
